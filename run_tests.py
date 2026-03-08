import subprocess
with open('test_output.txt', 'w') as f:
    subprocess.run(['venv\\Scripts\\pytest', 'tests/', '-v', '--tb=short'], stdout=f, stderr=f)
