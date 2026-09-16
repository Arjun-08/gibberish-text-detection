import subprocess,sys
steps=[("Dataset",[sys.executable,"-m","src.data_generator"]),("Model A",[sys.executable,"-m","src.train_statistical"]),("Model B",[sys.executable,"-m","src.train_classical"]),("Common evaluation",[sys.executable,"-m","src.evaluate_all"]),("Report",[sys.executable,"-m","src.report_generator"])]
for name,cmd in steps:
    print("\n"+"="*55+"\nRUNNING:",name,"\n"+"="*55); r=subprocess.run(cmd)
    if r.returncode: print("PIPELINE FAILED:",name); sys.exit(r.returncode)
print("\nPIPELINE COMPLETE. Run: streamlit run app.py")
