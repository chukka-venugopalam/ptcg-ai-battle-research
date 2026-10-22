import os, sys, json, time, traceback, multiprocessing as mp
from pathlib import Path
CAND='/mnt/data/m61_actual_out/agents/candidate'
BV1='/mnt/data/m61_actual_out/agents/Bv1'

def agent_worker(conn, agent_dir):
    try:
        os.chdir(agent_dir)
        sys.path.insert(0, agent_dir)
        ns={}
        code=compile(open(os.path.join(agent_dir,'main.py'),encoding='utf-8').read(), 'main.py','exec')
        exec(code, ns)  # intentionally no __file__, like Kaggle's exec mode
        agent=ns.get('agent')
        if not callable(agent): raise RuntimeError('agent callable not found')
        deck=agent({'select':None})
        conn.send({'ready':True,'deck_len':len(deck),'deck_unique':len(set(deck)),'deck':deck})
        while True:
            msg=conn.recv()
            if msg is None: break
            try:
                act=agent(msg)
                conn.send({'ok':True,'action':act})
            except Exception as e:
                conn.send({'ok':False,'error':type(e).__name__+': '+str(e),'tb':traceback.format_exc()})
        b=__import__('sa.bcagent',fromlist=['STATS'])
        conn.send({'stats':b.STATS.copy(),'health':b.health_line()})
    except Exception as e:
        try: conn.send({'ready':False,'error':type(e).__name__+': '+str(e),'tb':traceback.format_exc()})
        except Exception: pass
    finally:
        conn.close()

def main():
    ctx=mp.get_context('fork')
    procs=[]; pipes=[]
    for d in (BV1,CAND):
        a,b=ctx.Pipe()
        p=ctx.Process(target=agent_worker,args=(b,d),daemon=True);p.start();b.close();procs.append(p);pipes.append(a)
    ready=[]
    for i,p in enumerate(pipes):
        if not p.poll(120): raise RuntimeError(f'agent {i} startup timeout')
        r=p.recv();ready.append(r)
    print('AGENTS_READY',json.dumps([{k:v for k,v in r.items() if k!='deck'} for r in ready]))
    if not all(r.get('ready') for r in ready): raise RuntimeError('agent startup failed '+repr(ready))
    deck0,deck1=ready[0]['deck'],ready[1]['deck']
    # Use candidate R2 native SDK to play one real C-engine game. This is smoke-only, not BO3 statistics.
    sys.path.insert(0,CAND)
    from cg.game import battle_start, battle_select, battle_finish, Battle
    t0=time.time(); obs, sd=battle_start(deck0,deck1)
    if obs is None: raise RuntimeError(f'BattleStart rejected decks player={sd.errorPlayer} type={sd.errorType}')
    n=0; errors=[]; actor_counts=[0,0]; max_n=30000; sample_obs=[]
    while True:
        res=obs['current'].get('result',-1)
        if res != -1: break
        sel=obs.get('select')
        if n % 5 == 0 and sel is not None: sample_obs.append(obs)
        if sel is None: raise RuntimeError('select null before terminal result')
        actor=obs['current']['yourIndex']
        if actor not in (0,1): raise RuntimeError(f'invalid actor {actor}')
        pipes[actor].send(obs)
        if not pipes[actor].poll(30): raise RuntimeError(f'agent side {actor} action timeout at select {n}')
        ans=pipes[actor].recv()
        if not ans.get('ok'):
            raise RuntimeError(f'agent side {actor} exception at select {n}: {ans}')
        act=ans.get('action')
        opts=sel.get('option') or []
        if not isinstance(act,(list,tuple)) or not all(isinstance(x,int) for x in act):
            raise RuntimeError(f'non-integer action at select {n}: {act!r}')
        if len(act)<sel.get('minCount',0) or len(act)>sel.get('maxCount',0) or len(set(act))!=len(act) or any(x<0 or x>=len(opts) for x in act):
            raise RuntimeError(f'illegal action at select {n} side {actor}: action={act}, options={len(opts)}, bounds={sel.get("minCount")}-{sel.get("maxCount")}')
        obs=battle_select(list(act)); n+=1;actor_counts[actor]+=1
        if n>=max_n: raise RuntimeError('selection limit exceeded')
    outcome=obs['current']['result']
    json.dump(sample_obs, open('/mnt/data/m61_actual_out/real_engine_observations.json','w'))
    cardopts=sum(1 for o in sample_obs if any('cardId' in x or ('serial' in x and 'id' in x) for x in (o.get('select',{}).get('option') or []) if isinstance(x,dict)))
    atkopts=sum(1 for o in sample_obs if any('attackId' in x for x in (o.get('select',{}).get('option') or []) if isinstance(x,dict)))
    print('OBS_CAPTURE',json.dumps({'n':len(sample_obs),'card_option_observations':cardopts,'attack_option_observations':atkopts}))
    battle_finish()
    for p in pipes: p.send(None)
    stats=[]
    for p in pipes:
        if p.poll(30): stats.append(p.recv())
        else: stats.append({'stats':'not_received'})
    print('GAME_RESULT',json.dumps({'winner_player_index':outcome,'selects':n,'selects_by_seat':actor_counts,'elapsed_s':round(time.time()-t0,3),'stats':stats}))
    for p in procs:
        p.join(5)
        if p.is_alive(): p.terminate();p.join()
    print('NATIVE_ENGINE_SMOKE_OK')
if __name__=='__main__': main()
