#!/usr/bin/env python3
"""V2.2 host-overhead benchmark; no CUPTI required (uses CUDA Events + CPU clocks)."""
from __future__ import annotations
import argparse, csv, statistics, time
from pathlib import Path
import torch
import torch.nn.functional as F
import flash_attn_turing as ext
import flash_attention_interface as fi
from flash_attn_v22 import FlashAttnStaticGraph


def sync(device): torch.cuda.synchronize(device)

def host_submit_us(fn, device, calls=80, rounds=7, warmup=15):
    for _ in range(warmup): fn()
    sync(device); values=[]
    for _ in range(rounds):
        sync(device); t0=time.perf_counter_ns()
        for _ in range(calls): fn()
        t1=time.perf_counter_ns(); sync(device)
        values.append((t1-t0)/calls/1000.0)
    return statistics.median(values)

def event_us(fn, device, calls=40, rounds=7, warmup=10):
    for _ in range(warmup): fn()
    sync(device); values=[]
    for _ in range(rounds):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls): fn()
        end.record(); sync(device)
        values.append(start.elapsed_time(end)*1000.0/calls)
    return statistics.median(values)

def e2e_us(fn, device, trials=35, warmup=10):
    for _ in range(warmup): fn()
    sync(device); values=[]
    for _ in range(trials):
        t0=time.perf_counter_ns(); fn(); sync(device); t1=time.perf_counter_ns()
        values.append((t1-t0)/1000.0)
    return statistics.median(values)

def capture_graph(fn, n_ops, device):
    # Warm-up on side stream, as required by PyTorch CUDA Graph guidance.
    current=torch.cuda.current_stream(device); side=torch.cuda.Stream(device=device)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        for _ in range(3): fn()
    current.wait_stream(side); sync(device)
    graph=torch.cuda.CUDAGraph(); outputs=[]
    with torch.cuda.graph(graph):
        for _ in range(n_ops):
            outputs.append(fn())
    for _ in range(5): graph.replay()
    sync(device)
    return graph, outputs

def graph_metrics(fn, device, trials=40):
    host=[]; event=[]; wall=[]
    for _ in range(trials):
        sync(device)
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record(); t0=time.perf_counter_ns(); fn(); t1=time.perf_counter_ns()
        end.record(); sync(device); t2=time.perf_counter_ns()
        host.append((t1-t0)/1000.0)
        event.append(start.elapsed_time(end)*1000.0)
        wall.append((t2-t0)/1000.0)
    return statistics.median(host), statistics.median(event), statistics.median(wall)

def max_err(a,b): return (a.float()-b.float()).abs().max().item()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dims',type=int,nargs='+',default=[32,64,96,128])
    ap.add_argument('--seqs',type=int,nargs='+',default=[128,256,512])
    ap.add_argument('--batch',type=int,default=1); ap.add_argument('--heads',type=int,default=4)
    ap.add_argument('--causal',type=int,nargs='+',default=[0,1])
    ap.add_argument('--graph-ops',type=int,nargs='+',default=[1,4,16])
    ap.add_argument('--calls',type=int,default=80); ap.add_argument('--e2e-trials',type=int,default=35)
    ap.add_argument('--quick',action='store_true'); ap.add_argument('--output',type=Path,default=Path('v22_host_overhead.csv'))
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA is not available')
    if fi._ext_fwd_into is None: raise SystemExit('fwd_into is missing; rebuild the V2.2 extension first')
    if args.quick:
        args.dims=[32,64,96,128]; args.seqs=[128,256]; args.causal=[0]; args.graph_ops=[1,4]
    dev=torch.device('cuda',torch.cuda.current_device())
    print(f'# torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(dev)} cap={torch.cuda.get_device_capability(dev)}')
    print(f'# B={args.batch} H={args.heads} FP16 fastpath={fi._FASTPATH} fwd_into=enabled graph_ops={args.graph_ops}')
    fields=['kind','d','b','h','s','causal','candidate','n_ops','host_us','event_us','e2e_us','host_us_per_op','event_us_per_op','e2e_us_per_op','max_abs_error']
    writer=None; args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        def emit(row):
            writer.writerow(row); f.flush()
            print(','.join(str(row.get(k,'')) for k in fields),flush=True)
        for d in args.dims:
            for s in args.seqs:
                for causal_i in args.causal:
                    causal=bool(causal_i)
                    q,k,v=(torch.randn(args.batch,s,args.heads,d,device=dev,dtype=torch.float16) for _ in range(3))
                    qh,kh,vh=(x.transpose(1,2) for x in (q,k,v))
                    out,lse=fi.allocate_fwd_buffers(q); scale=d**-0.5
                    def sdpa(): return F.scaled_dot_product_attention(qh,kh,vh,is_causal=causal).transpose(1,2)
                    def legacy(): return fi.FlashAttnFunc.apply(q,k,v,scale,causal,False)
                    def public(): return fi.flash_attn_func(q,k,v,scale,causal)
                    def direct(): return ext.fwd(q,k,v,scale,causal)[0]
                    def into_raw(): ext.fwd_into(q,k,v,out,lse,scale,causal); return out
                    def into_checked(): return fi.flash_attn_func_into(q,k,v,out,lse,scale,causal)
                    # Warmup and correctness gate before recording timings.
                    refs=[sdpa(),legacy(),public(),direct(),into_raw(),into_checked()]
                    sync(dev); err=max(max_err(refs[1],refs[2]),max_err(refs[1],refs[3]),max_err(refs[1],refs[4]),max_err(refs[1],refs[5]))
                    if err>3e-3: raise RuntimeError(f'correctness mismatch D{d} S{s} causal={causal}: {err}')
                    for name,fn in [('sdpa',sdpa),('v2_function_apply',legacy),('v22_public_fastpath',public),('direct_ext_fwd_alloc',direct),('direct_fwd_into_raw',into_raw),('v22_checked_fwd_into',into_checked)]:
                        h=host_submit_us(fn,dev,args.calls); ev=event_us(fn,dev,args.calls); e=e2e_us(fn,dev,args.e2e_trials)
                        emit({'kind':'eager','d':d,'b':args.batch,'h':args.heads,'s':s,'causal':int(causal),'candidate':name,'n_ops':1,'host_us':round(h,3),'event_us':round(ev,3),'e2e_us':round(e,3),'host_us_per_op':round(h,3),'event_us_per_op':round(ev,3),'e2e_us_per_op':round(e,3),'max_abs_error':round(err,7)})
                    # N-operation graph: public fast path, fwd_into (one persistent out/LSE per op), and SDPA.
                    for n_ops in args.graph_ops:
                        for name,fn in [('sdpa',sdpa),('v22_public_fastpath',public)]:
                            graph,outputs=capture_graph(fn,n_ops,dev)
                            expected_graph = refs[0] if name == 'sdpa' else refs[1]
                            if max_err(outputs[-1], expected_graph) > 3e-3:
                                raise RuntimeError(f'graph correctness mismatch {name} D{d} S{s} causal={causal}')
                            gh,gev,ge=graph_metrics(graph.replay,dev)
                            emit({'kind':'graph','d':d,'b':args.batch,'h':args.heads,'s':s,'causal':int(causal),'candidate':name,'n_ops':n_ops,'host_us':round(gh,3),'event_us':round(gev,3),'e2e_us':round(ge,3),'host_us_per_op':round(gh/n_ops,3),'event_us_per_op':round(gev/n_ops,3),'e2e_us_per_op':round(ge/n_ops,3),'max_abs_error':round(err,7)})
                        # Separate output/LSE buffers for every op; stable addresses across replay.
                        bufs=[fi.allocate_fwd_buffers(q) for _ in range(n_ops)]
                        def into_n():
                            last=None
                            for ob,lb in bufs:
                                last=fi.flash_attn_func_into(q,k,v,ob,lb,scale,causal)
                            return last
                        graph,outputs=capture_graph(into_n,1,dev)  # into_n itself contains n_ops calls
                        if max_err(outputs[-1], refs[1]) > 3e-3:
                            raise RuntimeError(f'fwd_into graph correctness mismatch D{d} S{s} causal={causal}')
                        gh,gev,ge=graph_metrics(graph.replay,dev)
                        emit({'kind':'graph','d':d,'b':args.batch,'h':args.heads,'s':s,'causal':int(causal),'candidate':'v22_fwd_into','n_ops':n_ops,'host_us':round(gh,3),'event_us':round(gev,3),'e2e_us':round(ge,3),'host_us_per_op':round(gh/n_ops,3),'event_us_per_op':round(gev/n_ops,3),'e2e_us_per_op':round(ge/n_ops,3),'max_abs_error':round(err,7)})
                        # Separately measure the three input copies plus graph replay.
                        static_q=torch.empty_like(q); static_k=torch.empty_like(k); static_v=torch.empty_like(v)
                        static_q.copy_(q); static_k.copy_(k); static_v.copy_(v)
                        copy_bufs=[fi.allocate_fwd_buffers(q) for _ in range(n_ops)]
                        def static_into_n():
                            last=None
                            for ob,lb in copy_bufs:
                                last=fi.flash_attn_func_into(static_q,static_k,static_v,ob,lb,scale,causal)
                            return last
                        copy_graph,copy_outputs=capture_graph(static_into_n,1,dev)
                        if max_err(copy_outputs[-1], refs[1]) > 3e-3:
                            raise RuntimeError(f'copy-graph correctness mismatch D{d} S{s} causal={causal}')
                        def copy_then_replay():
                            static_q.copy_(q); static_k.copy_(k); static_v.copy_(v)
                            copy_graph.replay()
                        ch,cev,ce=graph_metrics(copy_then_replay,dev)
                        emit({'kind':'graph_copy','d':d,'b':args.batch,'h':args.heads,'s':s,'causal':int(causal),'candidate':'v22_fwd_into_copy_replay','n_ops':n_ops,'host_us':round(ch,3),'event_us':round(cev,3),'e2e_us':round(ce,3),'host_us_per_op':round(ch/n_ops,3),'event_us_per_op':round(cev/n_ops,3),'e2e_us_per_op':round(ce/n_ops,3),'max_abs_error':round(err,7)})
    print(f'# wrote {args.output.resolve()}')

if __name__=='__main__': main()
