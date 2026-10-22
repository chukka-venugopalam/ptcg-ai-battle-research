import os, sys, json, time, csv, traceback, multiprocessing as mp, argparse, hashlib
from pathlib import Path

ROOT=Path('/mnt/data/m61_work/agents')
AGENTS={
 'candidate': ROOT/'candidate',
 'bv1': ROOT/'bv1',
 'b2': ROOT/'b2',
 'bv2': ROOT/'bv2',
 'candidate_counter_true': ROOT/'candidate_counter_true',
 'candidate_shadowy2': ROOT/'candidate_shadowy2',
}

def worker(conn, agent_dir):
    try:
        os.chdir(agent_dir)
        sys.path.insert(0, str(agent_dir))
        code=compile(open(os.path.join(agent_dir,'main.py'),encoding='utf-8').read(), 'main.py','exec')
        ns={}
        exec(code, ns) # Kaggle contract: intentionally no __file__
        ag=ns.get('agent')
        if not callable(ag): raise RuntimeError('agent callable missing')
        deck=ag({'select':None})
        if not isinstance(deck,(list,tuple)) or len(deck)!=60: raise RuntimeError(f'bad deck handshake: {type(deck)} len={len(deck) if hasattr(deck,"__len__") else None}')
        conn.send({'ready':True,'deck':list(deck)})
        while True:
            msg=conn.recv()
            if msg is None: break
            try:
                action=ag(msg)
                conn.send({'ok':True,'action':action})
            except Exception as e:
                conn.send({'ok':False,'error':type(e).__name__+': '+str(e),'tb':traceback.format_exc()})
        try:
            from sa import bcagent as bc
            conn.send({'stats':dict(bc.STATS),'health':bc.health_line()})
        except Exception as e:
            conn.send({'stats_error':repr(e)})
    except Exception as e:
        try: conn.send({'ready':False,'error':type(e).__name__+': '+str(e),'tb':traceback.format_exc()})
        except Exception: pass
    finally:
        conn.close()

def start_worker(ctx, agent_name):
    a,b=ctx.Pipe()
    p=ctx.Process(target=worker,args=(b,str(AGENTS[agent_name])),daemon=True)
    p.start(); b.close()
    if not a.poll(90):
        p.terminate(); p.join(2)
        return None,None,{'ready':False,'error':'startup timeout'}
    r=a.recv()
    return p,a,r

def stop_worker(p, pipe):
    stats={}
    try:
        pipe.send(None)
        if pipe.poll(5): stats=pipe.recv()
    except Exception as e: stats={'cleanup_error':repr(e)}
    try:
        p.join(3)
        if p.is_alive(): p.terminate(); p.join(2)
    except Exception: pass
    try: pipe.close()
    except Exception: pass
    return stats

def play_game(ctx, agent0_name, agent1_name, game_id, pair_name):
    t0=time.time()
    # One fresh policy process per game, matching a per-episode agent lifecycle.
    p0, pipe0, r0 = start_worker(ctx, agent0_name)
    p1, pipe1, r1 = start_worker(ctx, agent1_name)
    row={'game_id':game_id,'pair':pair_name,'seat0_agent':agent0_name,'seat1_agent':agent1_name,
         'winner_index':None,'winner_agent':None,'selects':0,'selects_seat0':0,'selects_seat1':0,
         'illegal_actions':0,'action_errors':0,'engine_error':'','elapsed_s':None,
         'seat0_fallbacks':None,'seat1_fallbacks':None,'seat0_calls':None,'seat1_calls':None,
         'seat0_first_error':'','seat1_first_error':''}
    if not r0.get('ready') or not r1.get('ready'):
        row['engine_error']='agent startup failed: '+json.dumps({'seat0':{k:v for k,v in r0.items() if k!='deck'},'seat1':{k:v for k,v in r1.items() if k!='deck'}})
        if p0: stop_worker(p0,pipe0)
        if p1: stop_worker(p1,pipe1)
        row['elapsed_s']=round(time.time()-t0,4)
        return row
    deck0,deck1=r0['deck'],r1['deck']
    row['deck0_hash']=hashlib.sha256(json.dumps(deck0).encode()).hexdigest()
    row['deck1_hash']=hashlib.sha256(json.dumps(deck1).encode()).hexdigest()
    try:
        # Import R2 engine in parent after workers start, so workers don't inherit engine static state.
        agent_path=str(AGENTS['candidate'])
        if agent_path not in sys.path: sys.path.insert(0,agent_path)
        from cg.game import battle_start, battle_select, battle_finish
        obs, sd = battle_start(deck0,deck1)
        if obs is None:
            row['engine_error']=f'BattleStart rejected: player={sd.errorPlayer} type={sd.errorType}'
            return row
        n=0; seat_counts=[0,0]; errors=[]; max_selects=30000
        while True:
            cur=obs.get('current') or {}
            res=cur.get('result',-1)
            if res != -1:
                # Keep the engine-native winner code, including draw codes if any.
                row['winner_index']=int(res) if isinstance(res,(int,float)) else str(res)
                if row['winner_index'] in (0,1): row['winner_agent']=agent0_name if row['winner_index']==0 else agent1_name
                elif row['winner_index']==2: row['winner_agent']='draw_or_other_terminal_code_2'
                else: row['winner_agent']='unknown_terminal_code'
                break
            sel=obs.get('select')
            if sel is None: raise RuntimeError('select null while game still running')
            actor=cur.get('yourIndex')
            if actor not in (0,1): raise RuntimeError(f'invalid player index {actor}')
            pp=pipe0 if actor==0 else pipe1
            pp.send(obs)
            if not pp.poll(30): raise RuntimeError(f'action timeout on seat {actor} at selection {n}')
            ans=pp.recv()
            if not ans.get('ok'):
                row['action_errors']+=1
                raise RuntimeError(f'action exception seat={actor} select={n}: {ans.get("error")}\n{ans.get("tb","")}')
            act=ans.get('action')
            opts=sel.get('option') or []
            if not isinstance(act,(list,tuple)) or not all(isinstance(x,int) for x in act):
                row['illegal_actions']+=1; raise RuntimeError(f'non-int action: {act!r}')
            if len(act)<int(sel.get('minCount',0)) or len(act)>int(sel.get('maxCount',0)) or len(set(act))!=len(act) or any(x<0 or x>=len(opts) for x in act):
                row['illegal_actions']+=1; raise RuntimeError(f'illegal action seat={actor} action={act} options={len(opts)} bounds={sel.get("minCount")}-{sel.get("maxCount")}')
            obs=battle_select(list(act)); n+=1; seat_counts[actor]+=1
            if n>=max_selects: raise RuntimeError(f'exceeded {max_selects} selections')
        row['selects']=n; row['selects_seat0']=seat_counts[0]; row['selects_seat1']=seat_counts[1]
        battle_finish()
    except Exception as e:
        row['engine_error']=type(e).__name__+': '+str(e)
        row['traceback']=traceback.format_exc()[-5000:]
        try:
            from cg.game import battle_finish
            battle_finish()
        except Exception: pass
    finally:
        stats0=stop_worker(p0,pipe0) if p0 else {}
        stats1=stop_worker(p1,pipe1) if p1 else {}
        s0=stats0.get('stats',{}); s1=stats1.get('stats',{})
        row['seat0_fallbacks']=s0.get('fallbacks'); row['seat1_fallbacks']=s1.get('fallbacks')
        row['seat0_calls']=s0.get('calls'); row['seat1_calls']=s1.get('calls')
        row['seat0_first_error']=str(s0.get('first_error') or '')[:500]
        row['seat1_first_error']=str(s1.get('first_error') or '')[:500]
        row['elapsed_s']=round(time.time()-t0,4)
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--games',type=int,default=24,help='games per pairing, must be even to balance seat assignment')
    ap.add_argument('--pairs',default='candidate,bv1;candidate,b2;candidate,bv2;candidate,candidate_counter_true')
    ap.add_argument('--out',default='/mnt/data/m61_work/TOURNAMENT_GAMES_ACTUAL.csv')
    ap.add_argument('--checkpoint-every',type=int,default=2)
    args=ap.parse_args()
    if args.games<2 or args.games%2: raise SystemExit('--games must be even and >=2')
    pairs=[tuple(x.strip() for x in p.split(',')) for p in args.pairs.split(';') if p.strip()]
    for a,b in pairs:
        if a not in AGENTS or b not in AGENTS: raise SystemExit(f'unknown agent pair {a,b}; options={list(AGENTS)}')
    ctx=mp.get_context('fork')
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    fields=['game_id','pair','seat0_agent','seat1_agent','winner_index','winner_agent','selects','selects_seat0','selects_seat1','illegal_actions','action_errors','engine_error','elapsed_s','seat0_fallbacks','seat1_fallbacks','seat0_calls','seat1_calls','seat0_first_error','seat1_first_error','deck0_hash','deck1_hash','traceback']
    rows=[]; game_num=0; total_expected=len(pairs)*args.games; t_all=time.time()
    for pi,(a,b) in enumerate(pairs):
        for i in range(args.games):
            # Fully balanced player-index assignment; no engine RNG seed API exposed in this wrapper.
            seat0,seat1=(a,b) if i%2==0 else (b,a)
            game_num+=1
            row=play_game(ctx,seat0,seat1,game_num,f'{a}_vs_{b}')
            rows.append(row)
            errors=sum(bool(r.get('engine_error')) for r in rows)
            fall=sum((r.get('seat0_fallbacks') or 0)+(r.get('seat1_fallbacks') or 0) for r in rows)
            if game_num%args.checkpoint_every==0 or game_num==total_expected:
                print(f'PROGRESS {game_num}/{total_expected} pair={a}_vs_{b} elapsed={time.time()-t_all:.1f}s errors={errors} total_fallbacks={fall}',flush=True)
        print('PAIR_DONE',a,b,flush=True)
    # Persist the complete evidence atomically only after the run finishes.
    tmp = out.with_suffix(out.suffix + '.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    tmp.replace(out)
    print('CSV',str(out),flush=True)
    print('TOTAL_SECONDS',round(time.time()-t_all,2),flush=True)
    # Summary: game-level only, ties counted half for score rate.
    for a,b in pairs:
        rr=[r for r in rows if r['pair']==f'{a}_vs_{b}']
        wins=losses=ties=errors=0
        for r in rr:
            if r.get('engine_error'): errors+=1; continue
            if r.get('winner_agent')==a: wins+=1
            elif r.get('winner_agent')==b: losses+=1
            else: ties+=1
        n=len(rr)-errors
        print('SUMMARY',json.dumps({'pair':f'{a}_vs_{b}','games':len(rr),'valid_games':n,'wins_A':wins,'losses_A':losses,'ties':ties,'errors':errors,'A_score_rate':(wins+0.5*ties)/n if n else None}),flush=True)

if __name__=='__main__': main()
