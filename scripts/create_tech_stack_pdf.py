from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "tech_stack_summary.pdf"
FONT_REGULAR = Path("C:/Windows/Fonts/malgun.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/malgunbd.ttf")


SECTIONS = [
    (
        "Frontend",
        [
            "React 18",
            "Vite",
            "React Router",
            "Tailwind CSS",
            "lucide-react 아이콘",
            "Express 프록시 서버와 연동",
        ],
    ),
    (
        "Frontend Server",
        [
            "Node.js",
            "Express",
            "CORS",
            "concurrently",
            "`npm run dev:all`로 Vite 프론트와 Express 서버를 함께 실행",
        ],
    ),
    (
        "Backend / Analysis",
        [
            "Python 3.11",
            "Flask / Flask-CORS",
            "OpenCV",
            "NumPy",
            "scikit-learn",
            "FINCH clustering",
            "librosa / scipy / sounddevice",
        ],
    ),
    (
        "Speech / Language",
        [
            "OpenAI Whisper",
            "AI Hub KLUE-BERT 기반 언어 모델",
            "TensorFlow 2.15",
            "Hugging Face Transformers",
            "filler / repeat / pause 계열 언어 분석",
        ],
    ),
    (
        "Nonverbal / Vision",
        [
            "MediaPipe Holistic",
            "OpenFace 2.2.0",
            "STGCN++ / MMAction2",
            "PyTorch",
            "MMEngine / MMCV-lite",
        ],
    ),
    (
        "LLM Feedback",
        [
            "Ollama 연동 구조",
            "Qwen 계열 모델 우선 시도: qwen3:8b, qwen2.5:7b, qwen2.5:7b-instruct, qwen:7b",
            "일부 기존 코드에는 exaone3.5:7.8b 호출 구조도 존재",
            "Ollama 모델 미응답 시 fallback 피드백 생성",
        ],
    ),
    (
        "Data / Result",
        [
            "JSON 기반 결과 저장",
            "주요 출력: result/final_result.json, result/uploaded_session/final_result.json, result/session_history.json",
            "세션 인사이트: 발화 하이라이트, 상태 변화점, 반복 습관 후보, 전체 답변 피드백",
        ],
    ),
    (
        "Current Runtime URLs",
        [
            "Frontend: http://localhost:5173",
            "Express API: http://localhost:3001",
            "Flask video feed: http://localhost:5001/video_feed",
            "Latest result page: http://localhost:5173/result-latest",
        ],
    ),
]


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Malgun", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", str(FONT_BOLD)))


def build_pdf() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    register_fonts()

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleKo",
        parent=styles["Title"],
        fontName="Malgun-Bold",
        fontSize=20,
        leading=26,
        textColor=colors.HexColor("#111827"),
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "SubtitleKo",
        parent=styles["Normal"],
        fontName="Malgun",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12,
    )
    heading = ParagraphStyle(
        "HeadingKo",
        parent=styles["Heading2"],
        fontName="Malgun-Bold",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "BodyKo",
        parent=styles["BodyText"],
        fontName="Malgun",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="GTA 면접 분석 시스템 기술 스택",
    )

    story = [
        Paragraph("GTA 면접 분석 시스템 기술 스택", title),
        Paragraph("현재 프로젝트 코드와 설치된 모델/서버 구성을 기준으로 정리한 문서입니다.", subtitle),
    ]

    for section_title, items in SECTIONS:
        story.append(Paragraph(section_title, heading))
        bullets = [
            ListItem(Paragraph(item.replace("`", ""), body), leftIndent=8)
            for item in items
        ]
        story.append(
            ListFlowable(
                bullets,
                bulletType="bullet",
                start="circle",
                leftIndent=14,
                bulletFontName="Malgun",
                bulletFontSize=7,
            )
        )
        story.append(Spacer(1, 4))

    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build_pdf()
