"""
Submission Validation Script for FastSpeech 2 Take-Home Assignment (IIIT Hyderabad)
Runs automated integrity checks across folder structure, audios, plots, PDF report, and git state.
"""
import os
import sys
import glob

def check_structure():
    print("\n" + "="*60)
    print(" 1. FOLDER & FILE STRUCTURE CHECK")
    print("="*60)
    required = [
        ("English/train.py", "FILE"),
        ("English/preprocess.py", "FILE"),
        ("English/inference.py", "FILE"),
        ("English/config.yaml", "FILE"),
        ("English/checkpoints", "DIR"),
        ("Telugu/train.py", "FILE"),
        ("Telugu/preprocess.py", "FILE"),
        ("Telugu/inference.py", "FILE"),
        ("Telugu/config.yaml", "FILE"),
        ("Telugu/checkpoints", "DIR"),
        ("Results/English/training_loss.png", "FILE"),
        ("Results/English/validation_loss.png", "FILE"),
        ("Results/English/test_generated_audios", "DIR"),
        ("Results/English/variance_adaptor_plots/duration", "DIR"),
        ("Results/English/variance_adaptor_plots/pitch", "DIR"),
        ("Results/English/variance_adaptor_plots/energy", "DIR"),
        ("Results/Telugu/training_loss.png", "FILE"),
        ("Results/Telugu/validation_loss.png", "FILE"),
        ("Results/Telugu/test_generated_audios", "DIR"),
        ("Results/Telugu/variance_adaptor_plots/duration", "DIR"),
        ("Results/Telugu/variance_adaptor_plots/pitch", "DIR"),
        ("Results/Telugu/variance_adaptor_plots/energy", "DIR"),
        ("Report/Technical_Report.pdf", "FILE"),
        ("README.md", "FILE"),
        ("requirements.txt", "FILE"),
    ]
    all_ok = True
    for path, kind in required:
        exists = os.path.exists(path)
        if exists:
            print(f"  [PASS] {kind:<4} {path}")
        else:
            print(f"  [FAIL] {kind:<4} {path} (MISSING)")
            all_ok = False
    return all_ok

def check_audio():
    print("\n" + "="*60)
    print(" 2. AUDIO INTEGRITY CHECK (10 SAMPLES)")
    print("="*60)
    try:
        import soundfile as sf
    except ImportError:
        print("  [WARN] soundfile not installed, skipping acoustic reading.")
        return True

    expected_audios = [
        "Results/English/test_generated_audios/E1.wav",
        "Results/English/test_generated_audios/E2_a.wav",
        "Results/English/test_generated_audios/E2_b.wav",
        "Results/English/test_generated_audios/E3.wav",
        "Results/English/test_generated_audios/E4.wav",
        "Results/Telugu/test_generated_audios/T1.wav",
        "Results/Telugu/test_generated_audios/T2_a.wav",
        "Results/Telugu/test_generated_audios/T2_b.wav",
        "Results/Telugu/test_generated_audios/T3.wav",
        "Results/Telugu/test_generated_audios/T4.wav",
    ]
    all_ok = True
    for wav_path in expected_audios:
        if not os.path.exists(wav_path):
            print(f"  [FAIL] Missing audio: {wav_path}")
            all_ok = False
            continue
        try:
            data, sr = sf.read(wav_path)
            dur = len(data) / sr
            rms = float(data.std())
            peak = float(abs(data).max())
            status = "PASS" if (sr == 22050 and dur > 0.5 and rms > 0.05 and peak <= 1.0) else "WARN"
            if status == "WARN":
                all_ok = False
            print(f"  [{status}] {os.path.basename(wav_path):<10} | sr={sr}Hz | dur={dur:4.2f}s | rms={rms:.3f} | peak={peak:.3f}")
        except Exception as e:
            print(f"  [FAIL] Error reading {wav_path}: {e}")
            all_ok = False
    return all_ok

def check_plots():
    print("\n" + "="*60)
    print(" 3. VARIANCE ADAPTOR & LOSS PLOTS CHECK (34 PLOTS)")
    print("="*60)
    categories = [
        ("Results/English/training_loss.png", 1),
        ("Results/English/validation_loss.png", 1),
        ("Results/Telugu/training_loss.png", 1),
        ("Results/Telugu/validation_loss.png", 1),
        ("Results/English/variance_adaptor_plots/duration/*.png", 5),
        ("Results/English/variance_adaptor_plots/pitch/*.png", 5),
        ("Results/English/variance_adaptor_plots/energy/*.png", 5),
        ("Results/Telugu/variance_adaptor_plots/duration/*.png", 5),
        ("Results/Telugu/variance_adaptor_plots/pitch/*.png", 5),
        ("Results/Telugu/variance_adaptor_plots/energy/*.png", 5),
    ]
    all_ok = True
    for pattern, count in categories:
        matches = glob.glob(pattern) if "*" in pattern else ([pattern] if os.path.exists(pattern) else [])
        if len(matches) == count:
            print(f"  [PASS] {pattern:<55} ({len(matches)}/{count})")
        else:
            print(f"  [FAIL] {pattern:<55} ({len(matches)}/{count})")
            all_ok = False
    return all_ok

def check_report():
    print("\n" + "="*60)
    print(" 4. TECHNICAL REPORT INTEGRITY CHECK")
    print("="*60)
    pdf_path = "Report/Technical_Report.pdf"
    if not os.path.exists(pdf_path):
        print(f"  [FAIL] Report missing at: {pdf_path}")
        return False
    size_kb = os.path.getsize(pdf_path) / 1024
    if size_kb < 100:
        print(f"  [WARN] Report size unusually small: {size_kb:.1f} KB")
        return False

    pages = 0
    try:
        with open(pdf_path, "rb") as f:
            raw = f.read().decode("latin1", errors="ignore")
            pages = raw.count("/Type /Page\n") or raw.count("/Type /Page ") or raw.count("/Type/Page")
    except Exception:
        pass

    print(f"  [PASS] Technical_Report.pdf | Size: {size_kb:.1f} KB | Total Pages: {pages}")
    return True

def check_git():
    print("\n" + "="*60)
    print(" 5. GIT & GITHUB REPOSITORY STATUS")
    print("="*60)
    try:
        import subprocess
        status = subprocess.check_output(["git", "status", "-s"], text=True).strip()
        remote = subprocess.check_output(["git", "remote", "-v"], text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        
        print(f"  Branch: {branch}")
        print(f"  Remote: {remote.splitlines()[0] if remote else 'None'}")
        if status:
            print(f"  [WARN] Uncommitted changes:\n    " + "\n    ".join(status.splitlines()))
            return False
        else:
            print("  [PASS] Working tree clean, everything committed.")
            return True
    except Exception as e:
        print(f"  [WARN] Git check skipped: {e}")
        return True

def main():
    print("="*60)
    print(" FASTSPEECH 2 TAKE-HOME ASSIGNMENT: PRE-SUBMISSION AUDIT")
    print("="*60)
    r1 = check_structure()
    r2 = check_audio()
    r3 = check_plots()
    r4 = check_report()
    r5 = check_git()

    print("\n" + "="*60)
    print(" FINAL VERDICT")
    print("="*60)
    if r1 and r2 and r3 and r4 and r5:
        print("  >>> ALL CHECKS PASSED (100% READY FOR SUBMISSION) <<<")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        print("  >>> ISSUES DETECTED - PLEASE REVIEW FAILED CHECKS ABOVE <<<")
        print("="*60 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
