import subprocess
import os

# ระบุพาธของ Python interpreter ใน virtual environment
python_path = os.path.join(os.getcwd(), "env", "Scripts", "python.exe")

# ตรวจสอบ environment variables ที่ถูกตั้งค่าใน virtual environment
proc1 = subprocess.Popen([python_path, "Dashboard_for_sum_work_2025.py"],shell=True, env=os.environ.copy())
proc1

