"""
Creates 5 SYNTHETIC demo papers (fictional content) that match the PaperPilot pitch:
  P1-P3: stroke patients, continuous sEMG, the grip protocol works
  P4-P5: Parkinson's patients, single-point sensor, the grip protocol shows no effect
They exist only to test the full pipeline (a contradiction and a gap are planted on purpose).
They are NOT real research. Replace them with your real papers in data/papers/.
Run:  python scripts/generate_rehab_demo_pdfs.py
"""

import json
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rehab_demo_papers")

STROKE_RESULTS = (
    "Hand function scores improved significantly in stroke patients after the grip rehabilitation protocol. "
    "Muscle activation increased significantly during grip tasks measured with continuous sEMG. "
    "Grip strength also improved in stroke patients."
)
PD_RESULTS = (
    "Hand function scores did not improve in Parkinson's patients after the grip rehabilitation protocol. "
    "Grip strength showed no significant change in Parkinson's patients measured with a single-point sensor."
)

PAPERS = [
    {
        "file": "P1_Continuous_sEMG_Grip_Rehab_Stroke.pdf",
        "title": "Continuous sEMG-Guided Grip Rehabilitation in Chronic Stroke: A Pilot Study",
        "abstract": ("We show that a grip rehabilitation protocol guided by continuous sEMG feedback improved hand "
                     "function in stroke patients. All data were recorded in Dataset A."),
        "intro": "Hand weakness is common after stroke. Many rehabilitation protocols rely on strength scores only.",
        "methods": ("Participants were stroke patients treated in a single clinic. Continuous sEMG was recorded during "
                    "every grip training session and stored in Dataset A."),
        "results": STROKE_RESULTS,
        "disc": ("Limitations include a small sample and a single clinic. The protocol was not tested in other "
                 "patient groups."),
        "concl": "Continuous sEMG feedback supports recovery when the grip rehabilitation protocol is used.",
    },
    {
        "file": "P2_Surface_EMG_Biofeedback_Stroke_Trial.pdf",
        "title": "Surface EMG Biofeedback for Hand Recovery After Stroke: A Randomized Trial",
        "abstract": ("Results show that continuous sEMG biofeedback improved hand function in stroke patients who "
                     "followed the grip rehabilitation protocol. Data are in Dataset A."),
        "intro": "Biofeedback can guide effort during training. Evidence for its effect on hand function is limited.",
        "methods": "Stroke patients were randomised to the grip rehabilitation protocol. Continuous sEMG was recorded for all sessions.",
        "results": STROKE_RESULTS,
        "disc": "Limitations include short follow-up. The effect in other diseases has not been studied.",
        "concl": "The grip rehabilitation protocol with continuous sEMG is a promising option.",
    },
    {
        "file": "P3_Home_Based_Grip_Training_PostStroke.pdf",
        "title": "Home-Based Grip Training With Continuous sEMG in Post-Stroke Patients",
        "abstract": ("We demonstrate that home-based grip rehabilitation with continuous sEMG feedback improved hand "
                     "function in post-stroke patients. The data belong to Dataset A."),
        "intro": "Home training reduces clinic visits. Sensors can show patients how well they train.",
        "methods": "Post-stroke patients trained at home. A continuous sEMG sensor recorded each grip session in Dataset A.",
        "results": STROKE_RESULTS,
        "disc": "Limitations include self-reported training time. The protocol was not tested in other patient groups.",
        "concl": "Home-based grip training with continuous sEMG is feasible.",
    },
    {
        "file": "P4_Grip_Rehab_Parkinsons_SinglePoint.pdf",
        "title": "Grip Rehabilitation in Parkinson's Disease Using a Single-Point Sensor",
        "abstract": ("We find that the grip rehabilitation protocol did not improve hand function in Parkinson's "
                     "patients measured with a single-point sensor. Data were collected in Dataset B."),
        "intro": "Parkinson's disease affects hand control. Simple sensors are cheap and easy to use.",
        "methods": "Parkinson's patients followed the grip rehabilitation protocol. A single-point sensor on the forearm recorded activity in Dataset B.",
        "results": PD_RESULTS,
        "disc": ("The sensor records one point only, so the full time course of muscle activation was not captured. "
                 "Limitations include a small sample."),
        "concl": "The grip rehabilitation protocol needs a better measurement approach in Parkinson's disease.",
    },
    {
        "file": "P5_SinglePoint_EMG_Grip_Assessment_PD.pdf",
        "title": "Single-Point EMG Grip Assessment During Rehabilitation in Parkinson's Disease",
        "abstract": ("Results indicate that the grip rehabilitation protocol did not improve hand function in "
                     "Parkinson's patients assessed with a single-point sensor. Data are in Dataset B."),
        "intro": "Grip training is widely used. Its value in Parkinson's disease is unclear.",
        "methods": "Parkinson's patients completed the grip rehabilitation protocol. Muscle signals came from a single-point sensor and are stored in Dataset B.",
        "results": PD_RESULTS,
        "disc": "Limitations include a single centre. The protocol was not tested with other sensor types.",
        "concl": "More work is needed before the protocol can be recommended for Parkinson's disease.",
    },
]

GROUND_TRUTH = {
    "note": "Hand-written answer key for the SYNTHETIC rehab demo papers.",
    "claims": (
        [{"paper_id": f"P{i}", "population": "stroke", "method": "continuous_sEMG", "outcome": o, "direction": "positive"}
         for i in (1, 2, 3) for o in ("hand_function", "muscle_activation", "grip_strength")]
        + [{"paper_id": f"P{i}", "population": "parkinsons", "method": "single_point_sensor", "outcome": o, "direction": "none"}
           for i in (4, 5) for o in ("hand_function", "grip_strength")]
    ),
    "findings": [
        {"type": "contradiction", "paper_ids": ["P1", "P2", "P3", "P4", "P5"]},
        {"type": "gap", "population": "parkinsons", "method": "continuous_sEMG"},
        {"type": "gap", "population": "stroke", "method": "single_point_sensor"},
    ],
}


def build(p):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10.5, leading=15)
    head = ParagraphStyle("h", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=16, leading=20)
    small = ParagraphStyle("s", parent=ss["Normal"], fontSize=8, textColor="#888888")
    doc = SimpleDocTemplate(os.path.join(OUT, p["file"]), pagesize=A4, leftMargin=inch, rightMargin=inch,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch, title=p["title"])
    s = [Paragraph("SYNTHETIC DEMO PAPER. Fictional content created only to test PaperPilot.", small),
         Paragraph(p["title"], title),
         Paragraph("Fictional Author A, Fictional Author B (Synthetic Demo Institute), 2026", body),
         Paragraph("Abstract", head), Paragraph(p["abstract"], body),
         Paragraph("1. Introduction", head), Paragraph(p["intro"], body),
         Paragraph("2. Methodology", head), Paragraph(p["methods"], body),
         Paragraph("3. Results", head), Paragraph(p["results"], body),
         Paragraph("4. Discussion and Limitations", head), Paragraph(p["disc"], body),
         Paragraph("5. Conclusion", head), Paragraph(p["concl"], body),
         Paragraph("References", head), Paragraph("[1] Fictional A. et al. Synthetic reference. 2024.", body)]
    doc.build(s)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for paper in PAPERS:
        build(paper)
    with open(os.path.join(OUT, "ground_truth.json"), "w", encoding="utf-8") as fh:
        json.dump(GROUND_TRUTH, fh, indent=2)
    print(f"Created {len(PAPERS)} synthetic PDFs and ground_truth.json in {OUT}")
