import os, sys, json, copy, time, multiprocessing as mp, traceback
BASE='/mnt/data/m61_actual_out/agents'
AGENTS={'candidate':f'{BASE}/candidate','Bv1':f'{BASE}/Bv1','B2':f'{BASE}/B2','Bv2':f'{BASE}/Bv2'}
OBS=json.load(open('/mnt/data/m61_actual_out/real_engine_observations.json'))

def rewrite_ids(o, card=None, attack=None):
    o=copy.deepcopy(o)
    def w(x):
        if isinstance(x,dict):
            if card is not None:
                if 'serial' in x and 'id' in x: x['id']=card
                if 'cardId' in x: x['cardId']=card
            if attack is not None and 'attackId' in x: x['attackId']=attack
            for v in x.values(): w(v)
        elif isinstance(x,list):
            for v in x:w(v)
    w(o); return o

def worker(conn, agent_dir):
    try:
        os.chdir(agent_dir); sys.path.insert(0,agent_dir); ns={}
        exec(compile(open(os.path.join(agent_dir,'main.py'),encoding='utf-8').read(),'main.py','exec'),ns)
        a=ns['agent']; a({'select':None})
        bc=__import__('sa.bcagent',fromlist=['STATS']); conn.send({'ready':True,'deck':60,'health':bc.health_line()})
        while True:
            msg=conn.recv()
            if msg is None: break
            before=bc.STATS['fallbacks']; missbefore=bc.STATS['net_missing']
            try:
                act=a(msg['obs'])
                s=msg['obs']['select']; opts=s.get('option') or []
                legal=(isinstance(act,(list,tuple)) and all(isinstance(x,int) for x in act) and len(set(act))==len(act) and s.get('minCount',0)<=len(act)<=s.get('maxCount',0) and all(0<=x<len(opts) for x in act))
                conn.send({'ok':True,'legal':bool(legal),'fallback_delta':bc.STATS['fallbacks']-before,'net_missing_delta':bc.STATS['net_missing']-missbefore,'action':list(act) if isinstance(act,(list,tuple)) else str(act),'first_error':bc.STATS.get('first_error')})
            except Exception as e: conn.send({'ok':False,'error':repr(e),'tb':traceback.format_exc()})
        conn.send({'stats':bc.STATS.copy(),'health':bc.health_line()})
    except Exception as e:
        try:conn.send({'ready':False,'error':repr(e),'tb':traceback.format_exc()})
        except:pass
    finally:conn.close()

def main():
    card_obs=next((o for o in OBS if any(isinstance(x,dict) and 'cardId' in x for x in (o.get('select',{}).get('option') or []))),None)
    atk_obs=next((o for o in OBS if any(isinstance(x,dict) and 'attackId' in x for x in (o.get('select',{}).get('option') or []))),None)
    print('SAMPLE_OBS',{'total':len(OBS),'card_obs':card_obs is not None,'attack_obs':atk_obs is not None})
    card1300=rewrite_ids(card_obs,card=1300) if card_obs else None; card1431=rewrite_ids(card_obs,card=1431) if card_obs else None
    # Force a one-of choice so the neural forward path is exercised; the sampled original skill choice auto-selects all 3 and bypasses the model.
    for q in (card1300,card1431):
        if q is not None: q['select']['minCount']=1; q['select']['maxCount']=1
    variants=[('raw_card_context',card_obs),('card_1300_network_path',card1300),('card_1431_network_path',card1431),('raw_attack_context',atk_obs),('attack_1600',rewrite_ids(atk_obs,attack=1600) if atk_obs else None),('attack_1755',rewrite_ids(atk_obs,attack=1755) if atk_obs else None)]
    out={}
    ctx=mp.get_context('fork')
    for name,d in AGENTS.items():
        pa,ch=ctx.Pipe(); p=ctx.Process(target=worker,args=(ch,d),daemon=True);p.start();ch.close()
        if not pa.poll(120): raise RuntimeError('worker startup timeout '+name)
        startup=pa.recv(); record={'startup':startup,'tests':{}}
        if not startup.get('ready'): out[name]=record; p.terminate();p.join();continue
        for label,obs in variants:
            if obs is None: continue
            pa.send({'obs':obs})
            if not pa.poll(30): raise RuntimeError(f'timeout {name} {label}')
            record['tests'][label]=pa.recv()
        pa.send(None)
        if pa.poll(10): record['final']=pa.recv()
        p.join(3)
        if p.is_alive():p.terminate();p.join()
        out[name]=record
    print(json.dumps(out,indent=2))
    json.dump(out,open('/mnt/data/m61_actual_out/ID_BOUNDARY_PROBE_NATIVE.json','w'),indent=2)
if __name__=='__main__':main()
