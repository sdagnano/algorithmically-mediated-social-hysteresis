"""Causal-design schematics and simulation-based feasibility screen.

The simulations are planning exercises on standardized cluster-window outcomes.
They are not a final sample-size determination; nuisance ranges must be locked
from a blinded pilot and the exact randomization algorithm rerun before a study.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import reportlab
from reportlab.lib.colors import Color, HexColor, black
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

BLUE = HexColor("#1769AA")
ORANGE = HexColor("#D86F1D")
GREEN = HexColor("#238463")
PURPLE = HexColor("#7651A8")
RED = HexColor("#B33A3A")
GREY = HexColor("#66707A")
LIGHT = HexColor("#D8DEE6")
PALE = HexColor("#F4F6F8")

FONT_ROOT = Path(reportlab.__file__).resolve().parent / "fonts"
pdfmetrics.registerFont(TTFont("DesignSans", FONT_ROOT / "Vera.ttf"))
pdfmetrics.registerFont(TTFont("DesignSans-Bold", FONT_ROOT / "VeraBd.ttf"))


def font(c: canvas.Canvas, size: float, bold: bool = False) -> None:
    c.setFont("DesignSans-Bold" if bold else "DesignSans", size)


def ar_noise(rng: np.random.Generator, n_rep: int, n_clusters: int, n_grid: int, rho: float) -> np.ndarray:
    eps = np.empty((n_rep, n_clusters, n_grid), dtype=float)
    eps[:, :, 0] = rng.normal(size=(n_rep, n_clusters))
    innovation = math.sqrt(max(1e-9, 1.0 - rho**2))
    for j in range(1, n_grid):
        eps[:, :, j] = rho * eps[:, :, j-1] + innovation * rng.normal(size=(n_rep, n_clusters))
    return eps


SHAPES = {
    "broad_hump": np.array([0.25, 0.70, 1.00, 0.70, 0.25]),
    "localized_threshold": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    "broad_plateau": np.array([0.70, 1.00, 1.00, 1.00, 0.70]),
}
OUTER_REPLICATES = 1000
RANDOMIZATION_DRAWS = 499


def _balanced_histories(rng: np.random.Generator, n_draws: int, n_clusters: int) -> np.ndarray:
    scores = rng.random((n_draws, n_clusters))
    order = np.argsort(scores, axis=1)
    assignment = -np.ones((n_draws, n_clusters), dtype=np.int8)
    rows = np.arange(n_draws)[:, None]
    assignment[rows, order[:, : n_clusters // 2]] = 1
    return assignment


def _balanced_factorial(rng: np.random.Generator, n_draws: int, n_clusters: int) -> np.ndarray:
    if n_clusters % 4:
        raise ValueError("factorial screen requires cluster counts divisible by four")
    scores = rng.random((n_draws, n_clusters))
    order = np.argsort(scores, axis=1)
    cells = np.empty((n_draws, n_clusters), dtype=np.int8)
    rows = np.arange(n_draws)[:, None]
    width = n_clusters // 4
    for cell in range(4):
        cells[rows, order[:, cell*width:(cell+1)*width]] = cell
    return cells


def _two_arm_statistics(y: np.ndarray, keep: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    keep2 = keep[None, :]
    positive = (assignments == 1) & keep2
    negative = (assignments == -1) & keep2
    npg = positive.sum(axis=1).astype(float)
    nmg = negative.sum(axis=1).astype(float)
    if np.any(npg < 2) or np.any(nmg < 2):
        raise ValueError("assignment leaves fewer than two observed clusters in an arm")
    sp = positive @ y; sm = negative @ y
    qp = positive @ (y*y); qm = negative @ (y*y)
    mp = sp / npg[:, None]; mm = sm / nmg[:, None]
    vp = (qp - sp*sp/npg[:, None]) / (npg[:, None]-1.0)
    vm = (qm - sm*sm/nmg[:, None]) / (nmg[:, None]-1.0)
    se = np.sqrt(np.maximum(vp/npg[:, None] + vm/nmg[:, None], 1e-12))
    return np.max(np.abs((mp-mm)/se), axis=1)


def _factorial_statistics(y: np.ndarray, keep: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    means=[]; variances=[]; counts=[]
    for cell in range(4):
        mask=(assignments==cell)&keep[None,:]
        count=mask.sum(axis=1).astype(float)
        if np.any(count < 2):
            raise ValueError("assignment leaves fewer than two observed clusters in a factorial cell")
        total=mask@y; square=mask@(y*y)
        means.append(total/count[:,None])
        variances.append((square-total*total/count[:,None])/(count[:,None]-1.0))
        counts.append(count)
    contrast=means[0]-means[1]-means[2]+means[3]
    se=np.sqrt(np.maximum(sum(v/n[:,None] for v,n in zip(variances,counts)),1e-12))
    return np.max(np.abs(contrast/se),axis=1)


def _valid_reassignments(generator, statistic, rng, y, keep, n_clusters, n_draws):
    batches=[]
    while sum(len(batch) for batch in batches) < n_draws:
        proposed=generator(rng,max(32,n_draws),n_clusters)
        try:
            values=statistic(y,keep,proposed)
            batches.append(values)
        except ValueError:
            for assignment in proposed:
                try:
                    batches.append(statistic(y,keep,assignment[None,:]))
                except ValueError:
                    continue
    return np.concatenate(batches)[:n_draws]


def _history_rejection(rng, n_clusters, effect, shape, cluster_share, rho, attrition, leakage, n_perm):
    while True:
        observed=_balanced_histories(rng,1,n_clusters)[0]
        keep=rng.random(n_clusters)>=attrition
        if np.sum(keep&(observed==1))>=2 and np.sum(keep&(observed==-1))>=2:
            break
    u=rng.normal(0.0,math.sqrt(cluster_share),size=(n_clusters,1))
    eps=ar_noise(rng,1,n_clusters,5,rho)[0]*math.sqrt(1.0-cluster_share)
    y=u+eps+observed[:,None]*(effect*(1.0-leakage)/2.0)*shape[None,:]
    observed_stat=float(_two_arm_statistics(y,keep,observed[None,:])[0])
    permuted=_valid_reassignments(_balanced_histories,_two_arm_statistics,rng,y,keep,n_clusters,n_perm)
    p_value=(1.0+float(np.sum(permuted>=observed_stat)))/(n_perm+1.0)
    return p_value<=0.05


def _history_weak_null_rejection(
    rng, n_clusters, heterogeneity, shape, cluster_share, rho, n_perm
):
    """Studentized reassignment test under a false sharp null but exact weak null.

    Half of the finite population has a positive treatment-effect vector and half
    has its negative.  Hence the cluster-average treatment effect is exactly zero
    at every policy point although individual cluster effects are nonzero.
    Attrition and leakage are disabled here to isolate the weak-null diagnostic.
    """
    if n_clusters % 2:
        raise ValueError("weak-null diagnostic requires an even cluster count")
    observed=_balanced_histories(rng,1,n_clusters)[0]
    keep=np.ones(n_clusters,dtype=bool)
    tau=np.concatenate((
        np.full(n_clusters//2,heterogeneity,dtype=float),
        np.full(n_clusters//2,-heterogeneity,dtype=float),
    ))
    rng.shuffle(tau)
    u=rng.normal(0.0,math.sqrt(cluster_share),size=(n_clusters,1))
    eps=ar_noise(rng,1,n_clusters,5,rho)[0]*math.sqrt(1.0-cluster_share)
    y=u+eps+observed[:,None]*(tau[:,None]/2.0)*shape[None,:]
    observed_stat=float(_two_arm_statistics(y,keep,observed[None,:])[0])
    permuted=_valid_reassignments(
        _balanced_histories,_two_arm_statistics,rng,y,keep,n_clusters,n_perm
    )
    p_value=(1.0+float(np.sum(permuted>=observed_stat)))/(n_perm+1.0)
    return p_value<=0.05


def _factorial_rejection(rng,n_clusters,interaction,shape,cluster_share,rho,attrition,n_perm):
    while True:
        observed=_balanced_factorial(rng,1,n_clusters)[0]
        keep=rng.random(n_clusters)>=attrition
        if all(np.sum(keep&(observed==cell))>=2 for cell in range(4)):
            break
    signs=np.array([1,-1,-1,1],dtype=float)[observed]
    u=rng.normal(0.0,math.sqrt(cluster_share),size=(n_clusters,1))
    eps=ar_noise(rng,1,n_clusters,5,rho)[0]*math.sqrt(1.0-cluster_share)
    y=u+eps+signs[:,None]*(interaction/4.0)*shape[None,:]
    observed_stat=float(_factorial_statistics(y,keep,observed[None,:])[0])
    permuted=_valid_reassignments(_balanced_factorial,_factorial_statistics,rng,y,keep,n_clusters,n_perm)
    p_value=(1.0+float(np.sum(permuted>=observed_stat)))/(n_perm+1.0)
    return p_value<=0.05


def rejection_rate(kind:str,n_clusters:int,effect:float,shape:np.ndarray,seed:int,n_rep:int=OUTER_REPLICATES,n_perm:int=RANDOMIZATION_DRAWS)->tuple[float,float]:
    rng=np.random.default_rng(seed)
    rejections=[]
    for _ in range(n_rep):
        if kind=="history":
            rejections.append(_history_rejection(rng,n_clusters,effect,shape,.10,.50,.10,.10,n_perm))
        else:
            rejections.append(_factorial_rejection(rng,n_clusters,effect,shape,.10,.50,.10,n_perm))
    rate=float(np.mean(rejections))
    return rate,float(math.sqrt(rate*(1-rate)/n_rep))


def weak_null_rejection_rate(
    n_clusters:int, heterogeneity:float, shape:np.ndarray, seed:int,
    n_rep:int=OUTER_REPLICATES, n_perm:int=RANDOMIZATION_DRAWS,
) -> tuple[float,float]:
    rng=np.random.default_rng(seed)
    rejections=[
        _history_weak_null_rejection(rng,n_clusters,heterogeneity,shape,.10,.50,n_perm)
        for _ in range(n_rep)
    ]
    rate=float(np.mean(rejections))
    return rate,float(math.sqrt(rate*(1-rate)/n_rep))


def simulate_grid() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    shape=SHAPES["broad_hump"]
    history=[]
    for n in (48,80,128,192,256,320):
        type1,_=rejection_rate("history",n,0.0,shape,270000+n)
        for effect in (.30,.40,.50):
            power,mcse=rejection_rate("history",n,effect,shape,271000+n+int(effect*100))
            history.append({"n_clusters":n,"standardized_max_gap":effect,"shape":"broad_hump","cluster_random_intercept_variance_share":.10,"serial_rho":.50,"attrition":.10,"leakage":.10,"power":power,"type1":type1,"mcse":mcse})
    interactions=[]
    for n in (64,128,192,256,384,512):
        type1,_=rejection_rate("factorial",n,0.0,shape,280000+n)
        for effect in (.40,.50,.60):
            power,mcse=rejection_rate("factorial",n,effect,shape,281000+n+int(effect*100))
            interactions.append({"n_clusters":n,"standardized_history_reset_interaction":effect,"shape":"broad_hump","cluster_random_intercept_variance_share":.10,"serial_rho":.50,"attrition":.10,"power":power,"type1":type1,"mcse":mcse})
    sensitivity=[]
    for name,candidate in SHAPES.items():
        for kind,n,effect in (("history",256,.40),("factorial",384,.50)):
            rate,mcse=rejection_rate(kind,n,effect,candidate,290000+len(sensitivity)*1009,n_rep=OUTER_REPLICATES)
            sensitivity.append({"design":kind,"n_clusters":n,"effect":effect,"shape":name,"power":rate,"mcse":mcse})
    weak_null=[]
    for amplitude in (.40,.80,1.20):
        rate,mcse=weak_null_rejection_rate(
            256,amplitude,shape,300000+int(amplitude*1000)
        )
        weak_null.append({
            "n_clusters":256,
            "heterogeneity_amplitude":amplitude,
            "finite_population_average_effect":0.0,
            "sharp_null":False,
            "shape":"broad_hump",
            "cluster_random_intercept_variance_share":.10,
            "serial_rho":.50,
            "attrition":0.0,
            "leakage":0.0,
            "rejection_rate":rate,
            "mcse":mcse,
        })
    return (
        pd.DataFrame(history),pd.DataFrame(interactions),
        pd.DataFrame(sensitivity),pd.DataFrame(weak_null),
    )


def box(c:canvas.Canvas,x:float,y:float,w:float,h:float,heading:str,body:list[str],color)->None:
    c.setFillColor(Color(color.red,color.green,color.blue,alpha=.10)); c.setStrokeColor(color); c.setLineWidth(1.0)
    c.roundRect(x,y,w,h,6,fill=1,stroke=1); font(c,6.6,True); c.setFillColor(black); c.drawString(x+8,y+h-14,heading)
    font(c,6.3)
    for i,line in enumerate(body): c.drawString(x+8,y+h-28-i*10,line)


def arrow(c:canvas.Canvas,x1:float,y1:float,x2:float,y2:float,color=GREY)->None:
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(.9); c.line(x1,y1,x2,y2)
    angle=math.atan2(y2-y1,x2-x1); size=5.5
    for offset in (.48,-.48):
        c.line(x2,y2,x2-size*math.cos(angle+offset),y2-size*math.sin(angle+offset))


def fig_causal_dag(path:Path)->None:
    width,height=7.1*inch,4.20*inch; c=canvas.Canvas(str(path),pagesize=(width,height),initialFontName="DesignSans")
    # Time-unrolled causal structure. W is explicitly pre-treatment.
    nodes={
        "W":(.38,3.27,.78,.48,"baseline W",GREY),"H":(1.45,3.27,.82,.48,"history H",BLUE),"E":(2.58,3.27,.90,.48,"mapped E",PURPLE),
        "K0":(.56,2.16,1.02,.52,"candidates Kt",GREEN),"D0":(1.88,2.16,1.00,.52,"slate Dt",ORANGE),"S1":(3.18,2.16,1.02,.52,"state St+1",BLUE),
        "K1":(4.50,2.16,1.06,.52,"candidates Kt+1",GREEN),"D1":(5.84,2.16,.92,.52,"slate Dt+1",ORANGE),"Y":(5.84,.88,.92,.52,"outcome Y",RED),
        "R":(3.18,.88,1.02,.52,"reset Rtr",PURPLE),
    }
    centers={}; bounds={}
    for key,(xi,yi,wi,hi,label,color) in nodes.items():
        x,y,w,h=xi*inch,yi*inch,wi*inch,hi*inch; centers[key]=(x+w/2,y+h/2); bounds[key]=(x,y,w,h)
        c.setFillColor(Color(color.red,color.green,color.blue,alpha=.11)); c.setStrokeColor(color); c.roundRect(x,y,w,h,5,fill=1,stroke=1)
        font(c,7.0,True); c.setFillColor(black); c.drawCentredString(x+w/2,y+h/2-2,label)
    def boundary_point(key,target):
        cx,cy=centers[key]; tx,ty=target; _,_,w,h=bounds[key]; dx,dy=tx-cx,ty-cy
        scale=1.0/max(abs(dx)/(w/2),abs(dy)/(h/2))
        return cx+dx*scale,cy+dy*scale
    def connect(a,b,color=GREY):
        start=boundary_point(a,centers[b]); end=boundary_point(b,centers[a]); arrow(c,*start,*end,color=color)
    for a,b in (("W","H"),("H","E"),("H","K0"),("E","K0"),("K0","D0"),("D0","S1"),("S1","K1"),("E","K1"),("K1","D1"),("D1","Y"),("S1","Y")): connect(a,b)
    connect("R","S1",RED)
    font(c,6.0); c.setFillColor(RED); c.drawString(3.79*inch,1.63*inch,"replaces the state equation")
    font(c,6.0); c.setFillColor(GREY)
    c.drawString(.48*inch,.38*inch,"Candidate-generation rule may be common while realized K remains history-dependent.")
    c.drawString(.48*inch,.19*inch,"Reset separation requires every surviving H-to-Y path in the mutilated time-unrolled graph to be intercepted.")
    c.showPage(); c.save()


def plot_power(c:canvas.Canvas,frame:pd.DataFrame,x0:float,y0:float,w:float,h:float,xcol:str,groupcol:str,colors:dict)->None:
    c.setFillColor(PALE); c.rect(x0,y0,w,h,fill=1,stroke=0)
    c.setStrokeColor(LIGHT); c.line(x0,y0+.8*h,x0+w,y0+.8*h)
    xmin,xmax=float(frame[xcol].min()),float(frame[xcol].max())
    for group,data in frame.groupby(groupcol):
        data=data.sort_values(xcol); xs=x0+(data[xcol].to_numpy()-xmin)/(xmax-xmin)*w; ys=y0+data.power.to_numpy()*h
        color=colors[float(group)]; c.setStrokeColor(color); c.setLineWidth(1.7)
        p=c.beginPath(); p.moveTo(float(xs[0]),float(ys[0]));
        for xx,yy in zip(xs[1:],ys[1:]): p.lineTo(float(xx),float(yy))
        c.drawPath(p); c.setFillColor(color)
        for xx,yy in zip(xs,ys): c.circle(float(xx),float(yy),1.8,fill=1,stroke=0)
    c.setStrokeColor(black); c.rect(x0,y0,w,h,fill=0,stroke=1); font(c,6.4); c.setFillColor(black)
    for value in sorted(frame[xcol].unique()): c.drawCentredString(x0+(value-xmin)/(xmax-xmin)*w,y0-10,str(int(value)))
    c.saveState(); c.translate(x0-22,y0+h/2); c.rotate(90); c.drawCentredString(0,0,"power"); c.restoreState()
    c.drawCentredString(x0+w/2,y0-22,"randomized clusters")


def fig_design(path:Path,history:pd.DataFrame,interactions:pd.DataFrame)->None:
    width,height=7.1*inch,5.05*inch; c=canvas.Canvas(str(path),pagesize=(width,height),initialFontName="DesignSans")
    stages=[
        ("Stage 0: pilot",["blind; lock alpha,","variance share;","leakage, adherence"],GREY),
        ("Stage 1: history",["mirrored cluster paths;","randomization test"],BLUE),
        ("Stage 2: rate null",["fresh rates; reject only","declared relaxation class"],ORANGE),
        ("Stage 3: restoration",["factorial M/L policies;","C recovery separate"],GREEN),
        ("Stage 4: replication",["minor loops, new maxima;","external replication"],PURPLE),
    ]
    y=3.68*inch; box_w=1.22*inch; gap=.12*inch; x=.28*inch
    for i,(heading,body,color) in enumerate(stages):
        box(c,x,y,box_w,.78*inch,heading,body,color)
        if i<len(stages)-1: arrow(c,x+box_w,y+.39*inch,x+box_w+gap-2,y+.39*inch)
        x+=box_w+gap
    font(c,8.0,True); c.setFillColor(black); c.drawString(.30*inch,4.72*inch,"a   Gatekept experimental program")

    plot_power(c,history,.68*inch,.65*inch,2.55*inch,2.20*inch,"n_clusters","standardized_max_gap",{.3:GREY,.4:ORANGE,.5:BLUE})
    font(c,7.0,True); c.drawString(.33*inch,3.03*inch,"b   Stage-1 history-effect feasibility")
    font(c,6.4); c.setFillColor(GREY); c.drawString(.83*inch,2.67*inch,"max gap: 0.30 / 0.40 / 0.50 SD")

    plot_power(c,interactions,4.05*inch,.65*inch,2.40*inch,2.20*inch,"n_clusters","standardized_history_reset_interaction",{.4:GREY,.5:GREEN,.6:PURPLE})
    font(c,7.0,True); c.setFillColor(black); c.drawString(3.70*inch,3.03*inch,"c   Stage-3 history-by-restoration feasibility")
    font(c,6.4); c.setFillColor(GREY); c.drawString(4.18*inch,2.67*inch,"interaction: 0.40 / 0.50 / 0.60 SD")
    font(c,6.1); c.drawCentredString(width/2,.16*inch,"Illustrative design prior: cluster random-intercept variance share 0.10, AR(1) 0.50, attrition 0.10; Stage 1 leakage 0.10.")
    c.showPage(); c.save()


def main()->None:
    for directory in (DATA,FIGURES,RESULTS): directory.mkdir(parents=True,exist_ok=True)
    history,interactions,shape_sensitivity,weak_null=simulate_grid()
    history.to_csv(DATA/"power_history_grid.csv",index=False); interactions.to_csv(DATA/"power_reset_grid.csv",index=False)
    shape_sensitivity.to_csv(DATA/"power_shape_sensitivity.csv",index=False)
    weak_null.to_csv(DATA/"power_weak_null_heterogeneity.csv",index=False)
    fig_causal_dag(FIGURES/"fig2_causal_dag.pdf")
    fig_design(FIGURES/"fig5_experimental_design.pdf",history,interactions)
    summary={
        "status":"simulation-based feasibility screen; not a final sample-size determination",
        "outer_replicates_per_scenario":OUTER_REPLICATES,
        "randomization_reassignments_per_dataset":RANDOMIZATION_DRAWS,
        "history_design":{"grid_points":5,"shape":[.25,.70,1,.70,.25],"allocation":"balanced complete randomization to two histories, independently redrawn in every dataset","cluster_random_intercept_variance_share":.10,"serial_rho":.50,"attrition":.10,"leakage":.10,"empty_arm_rule":"redraw the entire Monte Carlo dataset only if attrition leaves fewer than two observed clusters in an assigned arm","test":"dataset-level studentized max statistic compared with complete-history reassignment distribution"},
        "reset_design":{"factorial_cells":4,"shape":[.25,.70,1,.70,.25],"allocation":"balanced complete randomization to the 2x2 history-by-restoration cells, independently redrawn in every dataset","cluster_random_intercept_variance_share":.10,"serial_rho":.50,"attrition":.10,"empty_cell_rule":"redraw the entire Monte Carlo dataset only if attrition leaves fewer than two observed clusters in an assigned cell","test":"dataset-level studentized maximum history-by-reset interaction compared with factorial reassignment distribution"},
        "history_grid":history.to_dict(orient="records"),"reset_grid":interactions.to_dict(orient="records"),
        "shape_sensitivity":shape_sensitivity.to_dict(orient="records"),
        "weak_null_heterogeneity_sanity":{
            "purpose":"finite-population zero-average-effect diagnostic with a false sharp null; not a proof of weak-null validity",
            "allocation":"balanced complete randomization to two histories",
            "attrition":0.0,
            "leakage":0.0,
            "records":weak_null.to_dict(orient="records"),
        },
        "required_final_workflow":"Lock nuisance ranges from a blinded pilot and rerun the same dataset-level assignment-reassignment algorithm with at least 20,000 outer replicates and a preregistered randomization-draw budget per candidate design.",
    }
    (RESULTS/"power_analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({
        "status":summary["status"],"history_scenarios":len(history),
        "reset_scenarios":len(interactions),"weak_null_scenarios":len(weak_null),
    },indent=2))


if __name__=="__main__": main()
