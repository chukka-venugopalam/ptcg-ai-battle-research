import sys, os, json, importlib.util
out = sys.stdout
sys.stdout = sys.stderr
bundle = sys.argv[1]
os.chdir(bundle); sys.path.insert(0, bundle)
spec = importlib.util.spec_from_file_location("main_w", os.path.join(bundle, "main.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for line in sys.stdin:
    obs = json.loads(line)
    try:
        act = m.agent(obs)
        res = {"act": [int(x) for x in act]}
    except Exception as e:
        res = {"err": repr(e)}
    out.write(json.dumps(res) + "\n"); out.flush()
