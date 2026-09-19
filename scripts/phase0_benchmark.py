import os
import io
import time
import json
import glob
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from google import genai
import sympy
from sympy.parsing.latex import parse_latex

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env")

client = genai.Client(api_key=API_KEY)

# 12 representative pages across the available tests
PILOT_SELECTIONS = [
    {"pdf": "docs/test1.pdf", "page": 0, "desc": "Linear Algebra - Matrix multiplication, definitions (test1 p1)"},
    {"pdf": "docs/test1.pdf", "page": 1, "desc": "Linear Algebra - Matrix equations, handwritten computations (test1 p2)"},
    {"pdf": "docs/test1.pdf", "page": 2, "desc": "Linear Algebra - Matrix steps, handwritten proof (test1 p3)"},
    {"pdf": "docs/test2sol.pdf", "page": 0, "desc": "Official Solution - Hebrew explanations & math symbols (test2sol p1)"},
    {"pdf": "docs/test2sol.pdf", "page": 1, "desc": "Official Solution - Formulas & boxed answers (test2sol p2)"},
    {"pdf": "docs/test3sol.pdf", "page": 0, "desc": "Mathematical definitions & derivations (test3sol p1)"},
    {"pdf": "docs/test4.pdf", "page": 0, "desc": "Student exam - Multi-line proof, dense handwriting (test4 p1)"},
    {"pdf": "docs/test4.pdf", "page": 1, "desc": "Student exam - Equations and step-by-step algebra (test4 p2)"},
    {"pdf": "docs/test5.pdf", "page": 0, "desc": "Student exam - Layout with marginal notes & formulas (test5 p1)"},
    {"pdf": "docs/test6.pdf", "page": 0, "desc": "Dense handwritten Hebrew script with strikethrough (test6 p1)"},
    {"pdf": "docs/test7sol.pdf", "page": 0, "desc": "Solution manual - Matrix calculations & Hebrew text (test7sol p1)"},
    {"pdf": "docs/test8sol.pdf", "page": 0, "desc": "Formal mathematical proofs & diagrams (test8sol p1)"},
]

MODELS_TO_TEST = [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview"
]

TRANSCRIPTION_PROMPT = """You are an expert OCR and transcription engine specialized in handwritten Hebrew academic exams containing advanced mathematics.

Analyze this scanned exam page and perform high-fidelity transcription into structured JSON.
Rules:
1. Detect and preserve Hebrew prose accurately (RTL reading order).
2. Transcribe all mathematical expressions, equations, and variables into clean standard LaTeX enclosed in $...$ for inline or $$...$$ for display equations.
3. If handwriting has crossed-out/strikethrough text, mark it with [[deleted: ...]] or omit if clearly irrelevant.
4. Identify distinct questions or sections on the page (e.g. Question 1, part a, etc.).
5. Produce your output strictly as a valid JSON object with the following schema:
{
  "summary": "Brief 1-line description of the page content",
  "questions": [
    {
      "question_label": "e.g. Question 1 / שאלה 1 / Part א",
      "hebrew_transcription": "The Hebrew prose transcribed",
      "latex_formulas": ["list", "of", "raw", "latex", "formulas", "extracted", "e.g. \\det(A) = 0"],
      "full_text": "Combined full text of this section in proper reading order with embedded $latex$"
    }
  ],
  "legibility_assessment": "High / Medium / Low",
  "transcription_confidence": 0.95
}

Return ONLY the raw JSON object without markdown fences.
"""

def extract_and_save_pilot_images():
    output_dir = os.path.join("storage", "crops", "pilot")
    os.makedirs(output_dir, exist_ok=True)
    
    extracted = []
    for idx, item in enumerate(PILOT_SELECTIONS):
        pdf_path = item["pdf"]
        page_num = item["page"]
        img_filename = f"pilot_{idx+1:02d}_{os.path.basename(pdf_path).replace('.pdf', '')}_p{page_num+1}.png"
        img_path = os.path.join(output_dir, img_filename)
        
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        pix.save(img_path)
        
        extracted.append({
            "id": f"pilot_{idx+1:02d}",
            "pdf": pdf_path,
            "page": page_num,
            "description": item["desc"],
            "image_path": img_path,
            "image_filename": img_filename,
            "width": pix.width,
            "height": pix.height
        })
        print(f"Rendered [{idx+1}/{len(PILOT_SELECTIONS)}] {img_filename}")
    return extracted

def test_latex_parseability(formulas):
    if not formulas:
        return {"total": 0, "parsed": 0, "parse_rate": 1.0, "errors": []}
    
    parsed_count = 0
    errors = []
    for f in formulas:
        cleaned = f.strip().strip("$").strip()
        if not cleaned:
            continue
        try:
            # Attempt sympy parsing
            parse_latex(cleaned)
            parsed_count += 1
        except Exception as e:
            errors.append({"formula": cleaned, "error": str(e)[:100]})
            
    total = len([f for f in formulas if f.strip().strip("$")])
    rate = (parsed_count / total) if total > 0 else 1.0
    return {
        "total": total,
        "parsed": parsed_count,
        "parse_rate": round(rate, 3),
        "sample_errors": errors[:3]
    }

def run_benchmark():
    pilot_items = extract_and_save_pilot_images()
    benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models_evaluated": MODELS_TO_TEST,
        "items": []
    }
    
    print("\nStarting VLM Transcription Benchmark on Pilot Set...")
    
    for item in pilot_items:
        print(f"\nEvaluating Pilot Item: {item['id']} ({item['description']})")
        item_result = {
            "metadata": item,
            "model_runs": {}
        }
        
        with open(item["image_path"], "rb") as f:
            img_bytes = f.read()
            
        for model_name in MODELS_TO_TEST:
            print(f"  -> Calling {model_name}...")
            start_time = time.time()
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        genai.types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        TRANSCRIPTION_PROMPT
                    ]
                )
                latency = round(time.time() - start_time, 2)
                raw_text = response.text.strip()
                
                # Strip potential markdown fences if present
                clean_json_str = raw_text
                if clean_json_str.startswith("```json"):
                    clean_json_str = clean_json_str[7:]
                if clean_json_str.startswith("```"):
                    clean_json_str = clean_json_str[3:]
                if clean_json_str.endswith("```"):
                    clean_json_str = clean_json_str[:-3]
                clean_json_str = clean_json_str.strip()
                
                try:
                    parsed_data = json.loads(clean_json_str)
                except Exception as json_err:
                    parsed_data = {
                        "summary": "JSON Parse Error",
                        "raw_output": raw_text,
                        "questions": [],
                        "error": str(json_err)
                    }
                
                # Gather all extracted LaTeX formulas
                all_formulas = []
                for q in parsed_data.get("questions", []):
                    all_formulas.extend(q.get("latex_formulas", []))
                    
                latex_eval = test_latex_parseability(all_formulas)
                
                # Usage metadata
                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                candidate_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
                
                item_result["model_runs"][model_name] = {
                    "latency_sec": latency,
                    "prompt_tokens": prompt_tokens,
                    "candidate_tokens": candidate_tokens,
                    "latex_eval": latex_eval,
                    "parsed_transcription": parsed_data,
                    "success": True
                }
                print(f"     Latency: {latency}s | Formulas: {latex_eval['total']} (Parsed: {latex_eval['parsed']})")
                
            except Exception as e:
                latency = round(time.time() - start_time, 2)
                print(f"     Error with {model_name}: {e}")
                item_result["model_runs"][model_name] = {
                    "latency_sec": latency,
                    "error": str(e),
                    "success": False
                }
            time.sleep(1)  # Respect rate limits
            
        benchmark_results["items"].append(item_result)
        
    # Save results to storage/reports
    results_path = os.path.join("storage", "reports", "phase0_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
        
    print(f"\nBenchmark Complete! Results saved to {results_path}")
    return results_path

if __name__ == "__main__":
    run_benchmark()
