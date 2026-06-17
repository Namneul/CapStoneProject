import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

const BACKEND_DIR = path.resolve(__dirname, '..');
const CURRENT_SESSION_DIR = path.join(BACKEND_DIR, 'result', 'current_session');
const LLM_MODELS = ['exaone3.5:7.8b', 'qwen3:8b', 'qwen2.5:7b', 'qwen2.5:7b-instruct', 'qwen:7b'];
let pythonProcess = null;

const fallbackQuestions = (topic = '') => {
    const normalized = String(topic || '').trim();
    if (normalized.includes('React') || normalized.includes('프론트')) {
        return [
            '최근 프론트엔드 프로젝트에서 맡았던 역할과 가장 신경 쓴 부분을 설명해 주세요.',
            'React로 화면을 만들 때 성능이나 사용자 경험을 개선했던 경험이 있다면 말해 주세요.',
            '팀 프로젝트에서 프론트엔드 개발 중 의견이 갈렸던 상황을 어떻게 조율했나요?',
        ];
    }
    if (normalized.includes('발표')) {
        return [
            '발표 주제를 처음 듣는 사람도 이해할 수 있게 핵심을 설명해 주세요.',
            '발표 준비 과정에서 가장 설득력 있게 전달하려고 한 부분은 무엇인가요?',
            '예상 질문이나 반대 의견을 받았을 때 어떻게 대응할 계획인가요?',
        ];
    }
    if (normalized.includes('피치') || normalized.includes('아이디어')) {
        return [
            '제안하려는 아이디어의 문제의식과 핵심 가치를 설명해 주세요.',
            '이 아이디어가 기존 방식보다 나은 점은 무엇이라고 생각하나요?',
            '실제로 실행한다면 가장 먼저 검증해야 할 위험 요소는 무엇인가요?',
        ];
    }
    return [
        '본인의 경험 중 이번 상황과 가장 관련 있는 사례를 소개해 주세요.',
        '그 경험에서 본인이 맡은 역할과 문제를 해결한 과정을 설명해 주세요.',
        '비슷한 상황을 다시 만난다면 무엇을 더 개선해서 해보고 싶나요?',
    ];
};

const cleanQuestion = (value) => String(value || '')
    .replace(/^[\s\-\d.)"']+/, '')
    .replace(/["']/g, '')
    .trim();

const parseQuestions = (text) => {
    const trimmed = String(text || '').trim();
    try {
        const parsed = JSON.parse(trimmed);
        const list = Array.isArray(parsed) ? parsed : parsed.questions;
        if (Array.isArray(list)) {
            return list.map(cleanQuestion).filter(Boolean).slice(0, 3);
        }
    } catch (error) {
        // Fall through to line parsing.
    }
    return trimmed
        .split(/\r?\n/)
        .map(cleanQuestion)
        .filter((line) => line.endsWith('?') || line.endsWith('요.'))
        .slice(0, 3);
};

const parseSessionMeta = (value) => {
    if (!value) return null;
    try {
        const parsed = JSON.parse(String(value));
        if (!Array.isArray(parsed.answers)) return null;
        let cursor = 0;
        const answers = parsed.answers.map((answer, index) => {
            const duration = Number(answer.duration || 0);
            const start = cursor;
            const end = cursor + Math.max(duration, 0);
            cursor = end;
            return {
                index: index + 1,
                question: String(answer.question || `질문 ${index + 1}`),
                duration,
                start,
                end,
                completedAt: answer.completedAt || null,
            };
        });
        return {
            enabled: true,
            topic: String(parsed.topic || ''),
            answers,
            totalDuration: cursor,
        };
    } catch (error) {
        return null;
    }
};

const callLocalLlm = async (prompt) => {
    for (const model of LLM_MODELS) {
        try {
            const response = await fetch('http://localhost:11434/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model,
                    prompt,
                    stream: false,
                    temperature: 0.35,
                }),
            });
            if (!response.ok) continue;
            const data = await response.json();
            const answer = String(data.response || '').trim();
            if (answer) return answer;
        } catch (error) {
            continue;
        }
    }
    return '';
};

app.post('/api/generate-questions', async (req, res) => {
    const topic = String(req.body?.topic || '').trim() || '일반 면접 연습';
    const prompt = `너는 한국어 면접 코치다.
아래 연습 주제에 맞는 실제 면접 질문 3개를 만들어라.
질문은 억지로 주제명을 끼워 넣지 말고, 면접관이 자연스럽게 물어볼 법한 문장으로 작성해라.
각 질문은 서로 다른 평가 포인트를 다뤄야 한다.
반드시 JSON 배열만 출력해라. 예: ["질문1", "질문2", "질문3"]

[연습 주제]
${topic}`;

    const llmAnswer = await callLocalLlm(prompt);
    const questions = parseQuestions(llmAnswer);
    res.json({
        questions: questions.length >= 3 ? questions : fallbackQuestions(topic),
        source: questions.length >= 3 ? 'llm' : 'fallback',
    });
});

app.post('/api/start-interview', (req, res) => {
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
    }

    const { topic } = req.body;
    let topicName = topic || '일반 면접 연습';

    // 한글 경로(문서)로 인한 Mediapipe C++ 코어 버그 방지를 위해 심볼릭 링크 경로 사용
    const localPython = path.join(BACKEND_DIR, '.venv', 'Scripts', 'python.exe');
    const pythonExePath = fs.existsSync(localPython) ? localPython : 'python';
    pythonProcess = spawn(pythonExePath, ['-u', 'orchestrator.py'], {
        cwd: BACKEND_DIR,
        env: {
            ...process.env,
            PYTHONUNBUFFERED: "1",
            PYTHONIOENCODING: "utf-8",
            PATH: `${path.dirname(pythonExePath)};${process.env.PATH}`,
        }
    });
    
    let question = "";
    let isQuestionStarted = false;
    let isReadyForAnswer = false;

    const currentProcess = pythonProcess;

    currentProcess.stdout.on('data', (data) => {
        const text = data.toString();
        console.log(`[Python]: ${text}`);
        
        if (text.includes('번호를 선택하세요:')) {
            // pass '4' to trigger custom input if it doesn't match 1-3 exactly
            let topicInput = '4\n';
            if (topic === '1' || topic === '2' || topic === '3') {
                topicInput = `${topic}\n`;
            }
            currentProcess.stdin.write(topicInput);
        } else if (text.includes('상황을 직접 입력해주세요:')) {
            currentProcess.stdin.write(`${topicName}\n`);
        } else if (text.includes('Enter 누르면 시작')) {
            // start the interview to get the question
            currentProcess.stdin.write('\n');
        } else if (text.includes('질문:')) {
            isQuestionStarted = true;
            const parts = text.split('질문:');
            if (parts.length > 1) {
                question += parts[1];
            }
        } else if (isQuestionStarted && !text.includes('답변 준비되면 Enter 누르세요')) {
            question += text;
        }

        if (text.includes('답변 준비되면 Enter 누르세요')) {
            isQuestionStarted = false;
            isReadyForAnswer = true;
            // Return the question to the frontend!
            if (!res.headersSent) {
                res.json({ question: question.trim() });
            }
        }
    });

    currentProcess.stderr.on('data', (data) => {
        console.error(`[Python Error]: ${data}`);
    });
    
    currentProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        if (pythonProcess === currentProcess) {
            pythonProcess = null;
        }
        // if it crashes before sending question
        if (!res.headersSent) {
            res.status(500).json({ error: "Failed to generate question" });
        }
    });
});

app.post('/api/start-recording', (req, res) => {
    if (!pythonProcess) {
        return res.status(400).json({ error: "No active interview session." });
    }

    const currentProcess = pythonProcess;

    // Send enter to start recording/camera
    currentProcess.stdin.write('\n');
    
    // Wait for process to finish
    currentProcess.on('close', (code) => {
        if (pythonProcess === currentProcess) {
            pythonProcess = null;
        }
        // read result/final_result.json
        const resultPath = path.join(BACKEND_DIR, 'result', 'final_result.json');
        if (fs.existsSync(resultPath)) {
            try {
                const data = fs.readFileSync(resultPath, 'utf8');
                res.json(JSON.parse(data));
            } catch (e) {
                res.status(500).json({ error: "Result file is invalid JSON" });
            }
        } else {
            res.status(500).json({ error: "Result file not found" });
        }
    });
});

app.post('/api/stop-recording', (req, res) => {
    if (pythonProcess && pythonProcess.stdin) {
        // Send 'q' to stop recording
        pythonProcess.stdin.write('q\n');
        res.json({ success: true });
    } else {
        res.status(400).json({ error: "No active process to stop" });
    }
});

app.post('/api/cancel-interview', (req, res) => {
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
    }
    res.json({ success: true });
});

app.get('/api/latest-result', (req, res) => {
    const resultPath = path.join(BACKEND_DIR, 'result', 'final_result.json');
    if (!fs.existsSync(resultPath)) {
        return res.status(404).json({ error: "Result file not found" });
    }
    try {
        const data = fs.readFileSync(resultPath, 'utf8');
        res.json(JSON.parse(data));
    } catch (e) {
        res.status(500).json({ error: "Result file is invalid JSON" });
    }
});

app.get('/api/current-result', (req, res) => {
    const resultPath = path.join(CURRENT_SESSION_DIR, 'final_result.json');
    if (!fs.existsSync(resultPath)) {
        return res.status(404).json({ error: "Current session result file not found" });
    }
    try {
        const data = fs.readFileSync(resultPath, 'utf8');
        res.json(JSON.parse(data));
    } catch (e) {
        res.status(500).json({ error: "Current session result file is invalid JSON" });
    }
});

app.post('/api/analyze-current-session', express.raw({ type: 'application/octet-stream', limit: '500mb' }), (req, res) => {
    const topic = String(req.query.topic || '면접 연습');
    const question = String(req.query.question || '현재 진행한 면접 연습 전체 답변입니다. 답변 흐름과 전달 방식을 분석하세요.');
    const sessionMeta = parseSessionMeta(req.query.session);

    if (!req.body || !req.body.length) {
        return res.status(400).json({ error: '녹화된 영상 데이터가 없습니다.' });
    }

    fs.mkdirSync(CURRENT_SESSION_DIR, { recursive: true });
    const webmPath = path.join(CURRENT_SESSION_DIR, 'recording.webm');
    const mp4Path = path.join(CURRENT_SESSION_DIR, 'recording.mp4');
    fs.writeFileSync(webmPath, req.body);

    const ffmpeg = spawn('ffmpeg', [
        '-y',
        '-i', webmPath,
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        mp4Path,
    ], { cwd: BACKEND_DIR });

    let ffmpegError = '';
    ffmpeg.stderr.on('data', (data) => {
        ffmpegError += data.toString();
    });

    ffmpeg.on('close', (code) => {
        if (code !== 0 || !fs.existsSync(mp4Path)) {
            return res.status(500).json({ error: '녹화 영상을 분석 가능한 형식으로 변환하지 못했습니다.', detail: ffmpegError.slice(-1000) });
        }

        const localPython = path.join(BACKEND_DIR, '.venv', 'Scripts', 'python.exe');
        const pythonExePath = fs.existsSync(localPython) ? localPython : 'python';
        const analyzer = spawn(pythonExePath, [
            'analyze_uploaded_session.py',
            mp4Path,
            '--question', question,
            '--situation', topic,
            '--output-dir', CURRENT_SESSION_DIR,
        ], {
            cwd: BACKEND_DIR,
            env: {
                ...process.env,
                PYTHONUNBUFFERED: '1',
                PYTHONIOENCODING: 'utf-8',
                PATH: `${path.dirname(pythonExePath)};${process.env.PATH}`,
            },
        });

        let stderr = '';
        analyzer.stderr.on('data', (data) => {
            stderr += data.toString();
            console.error(`[Current Session Analyzer Error]: ${data}`);
        });
        analyzer.stdout.on('data', (data) => {
            console.log(`[Current Session Analyzer]: ${data}`);
        });
        analyzer.on('close', (analyzerCode) => {
            const resultPath = path.join(CURRENT_SESSION_DIR, 'final_result.json');
            if (analyzerCode !== 0 || !fs.existsSync(resultPath)) {
                return res.status(500).json({
                    error: '현재 세션 분석에 실패했습니다.',
                    detail: stderr.slice(-1200),
                });
            }
            try {
                const data = JSON.parse(fs.readFileSync(resultPath, 'utf8'));
                if (sessionMeta) {
                    data.frontend_session = sessionMeta;
                    fs.writeFileSync(resultPath, JSON.stringify(data, null, 2), 'utf8');
                    fs.writeFileSync(path.join(BACKEND_DIR, 'result', 'final_result.json'), JSON.stringify(data, null, 2), 'utf8');
                }
                res.json(data);
            } catch (error) {
                res.status(500).json({ error: '현재 세션 결과 파일을 읽지 못했습니다.' });
            }
        });
    });
});

const PORT = 3001;
app.listen(PORT, () => {
    console.log(`Proxy server running on http://localhost:${PORT}`);
});
