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
let pythonProcess = null;

app.post('/api/start-interview', (req, res) => {
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
    }

    const { topic } = req.body;
    let topicName = topic || '일반 면접 연습';

    // 한글 경로(문서)로 인한 Mediapipe C++ 코어 버그 방지를 위해 심볼릭 링크 경로 사용
    const pythonExePath = 'C:\\Users\\exlle\\CapStone_venv\\Scripts\\python.exe';
    pythonProcess = spawn(pythonExePath, ['-u', 'orchestrator.py'], {
        cwd: path.join(__dirname, '..'), // root 디렉토리에서 실행
        env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8", PATH: `C:\\Users\\exlle\\CapStone_venv\\Scripts;${process.env.PATH}` }
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

const PORT = 3001;
app.listen(PORT, () => {
    console.log(`Proxy server running on http://localhost:${PORT}`);
});
