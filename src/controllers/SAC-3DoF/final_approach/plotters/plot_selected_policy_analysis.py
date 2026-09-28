"""Plot evaluation results for selected final-approach SAC policies."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
from envs.docking3dof.gym_env import Docking3DoFGymEnv

HERE = Path(__file__).resolve().parents[1]
MODELS = HERE / "models"
RESULTS = HERE / "results"
OUT = RESULTS / "retained_figures" / "selected_policy_analysis"
LEARNING_STARTS = 10_000
MAX_STEPS = 5_000
BLUE, DARK, GREEN, RED, GREY = "#0072B2", "#003B5C", "#009E73", "#D55E00", "#777777"

TRAINING = [
    (5, 3, 600_000, 600_000, 100_000, "best"),
    (4, 10, 600_000, 100_000, 570_000, "best"),
    (5, 11, 600_000, 600_000, 600_000, "final"),
]


def stem(seed, budget, buffer_size, kind):
    value = f"sac_final_approach_portobs_rewardv9_{budget//1000}ksteps_seed{seed}_optuna12"
    if budget != 100_000:
        value += f"_buffer{buffer_size//1000}k"
    return value + ("_best" if kind == "best" else "")


def group(budget, buffer_size):
    return f"optuna12_{budget//1000}ksteps_buffer{buffer_size//1000}k"


def campaign_directory(root, budget, buffer_size):
    """Return the retained or development directory for a campaign."""

    campaign = group(budget, buffer_size)
    category = (
        "retained_policy_evaluations"
        if root == RESULTS and campaign in {
            "optuna12_100ksteps_buffer100k",
            "optuna12_600ksteps_buffer600k",
        }
        else "development_experiments"
        if root == RESULTS
        else "retained_policies"
        if campaign in {
            "optuna12_100ksteps_buffer100k",
            "optuna12_600ksteps_buffer600k",
        }
        else "development_experiments"
    )
    return root / category / campaign


def style():
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9.5, "figure.dpi": 120, "savefig.dpi": 300})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def checkpoint_episode(data, checkpoint):
    i = min(int(np.searchsorted(data.timestep.to_numpy(), checkpoint, side="left")), len(data)-1)
    return int(data.episode.iloc[i]), int(data.timestep.iloc[i])


def training_plots():
    records = []
    for rnd, seed, budget, buf, checkpoint, kind in TRAINING:
        run_stem = stem(seed, budget, buf, "final")
        path = campaign_directory(RESULTS, budget, buf) / "learning_curves" / run_stem / "learning_curve.csv"
        data = pd.read_csv(path)
        start_i = int(np.flatnonzero(data.timestep.to_numpy() >= LEARNING_STARTS)[0])
        shown = data.iloc[start_i:].copy()
        selected_episode, boundary_step = checkpoint_episode(data, checkpoint)
        rolling = shown.episode_return.rolling(10, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(shown.episode, shown.episode_return, color=BLUE, lw=.8, alpha=.28, label="Individual episode")
        ax.plot(shown.episode, rolling, color=DARK, lw=2, label="10-episode moving average")
        ax.axvline(selected_episode, color=RED, ls="--", lw=1.5,
                   label=f"Selected {kind} policy ({checkpoint//1000}k steps; episode {selected_episode})")
        revise_typography = (rnd, seed) in {(4, 10), (5, 11)}
        ax.set(xlabel="Episode number" if revise_typography else "Training episode number",
               ylabel="Episode return")
        ax.set_title(f"SAC final-approach training — Round {rnd}, seed {seed}", loc="left",
                     fontweight="normal" if revise_typography else "bold")
        ax.grid(True, color="#D9D9D9", lw=.65, alpha=.75); ax.set_axisbelow(True); ax.margins(x=.01)
        ax.legend(frameon=False, loc="best")
        fig.subplots_adjust(left=.12, right=.985, bottom=.14, top=.90)
        save(fig, f"r{rnd}_s{seed}_b{buf//1000}k_learning_curve")

        if rnd == 5 and seed == 3:
            durations = data.timestep.diff().fillna(data.timestep).astype(float)
            shown_duration = durations.iloc[start_i:]
            smooth_duration = shown_duration.rolling(10, min_periods=1).mean()
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(shown.episode, shown_duration, color=BLUE, lw=.8, alpha=.28, label="Individual episode")
            ax.plot(shown.episode, smooth_duration, color=DARK, lw=2, label="10-episode moving average")
            ax.axvline(selected_episode, color=RED, ls="--", lw=1.5,
                       label=f"Selected checkpoint (100k steps; episode {selected_episode})")
            ax.set(xlabel="Episode number", ylabel="Episode duration [s]")
            ax.set_title("SAC episode duration — Round 5, seed 3", loc="left", fontweight="normal")
            ax.grid(True, color="#D9D9D9", lw=.65, alpha=.75); ax.set_axisbelow(True); ax.margins(x=.01)
            ax.legend(frameon=False, loc="best")
            fig.subplots_adjust(left=.12, right=.985, bottom=.14, top=.90)
            save(fig, "r5_s3_b600k_episode_duration")
        records.append({"round": rnd, "seed": seed, "budget": budget, "buffer": buf,
            "checkpoint_step": checkpoint, "checkpoint_type": kind,
            "checkpoint_episode": selected_episode, "episode_boundary_timestep": boundary_step,
            "trim_start_episode": int(shown.episode.iloc[0]), "trim_start_timestep": int(shown.timestep.iloc[0]),
            "training_file": str(path)})
    return records


def make_env(seed):
    return Docking3DoFGymEnv(phase="final", mode="testing", dt=1., max_time=5_000.0,
        integrator="rk4", normalize_action=True, seed=seed)


def load_pair(seed, budget, buf, kind):
    name = stem(seed, budget, buf, kind)
    folder = campaign_directory(MODELS, budget, buf)
    mp, vp = folder / f"{name}.zip", folder / f"{name}_vecnormalize.pkl"
    dummy = DummyVecEnv([lambda: Monitor(make_env(999))])
    vn = VecNormalize.load(str(vp), dummy); vn.training = False; vn.norm_reward = False
    return SAC.load(str(mp)), vn, dummy, mp, vp


def rollout(model, vn, seed):
    env = make_env(seed); obs, _ = env.reset(seed=seed)
    q = env.core_env.get_derived_quantities(env.core_env.state)
    h = {"time":[0.], "x":[float(q["r_p"][0])], "y":[float(q["r_p"][1])],
         "pos":[float(np.linalg.norm(q["r_p"]))], "vel":[float(np.linalg.norm(q["v_p"]))],
         "att":[abs(float(np.rad2deg(q["psi_e"])))], "rate":[abs(float(np.rad2deg(q["omega"])))],
         "s":[float(q["s"])], "rho":[float(q["rho"])]}
    info = {}
    for _ in range(MAX_STEPS):
        a, _ = model.predict(vn.normalize_obs(obs.reshape(1,-1)), deterministic=True)
        obs, _, term, trunc, info = env.step(a[0])
        h["time"].append(float(env.core_env.time)); h["x"].append(float(info["port_relative_position"][0])); h["y"].append(float(info["port_relative_position"][1]))
        h["pos"].append(float(info["port_distance"])); h["vel"].append(float(info["port_speed"])); h["att"].append(abs(float(np.rad2deg(info["psi_e"])))); h["rate"].append(abs(float(np.rad2deg(info["omega"])))); h["s"].append(float(info["s"])); h["rho"].append(float(info["rho"]))
        if term or trunc: break
    env.close()
    return {k:np.asarray(v) for k,v in h.items()} | {"success":bool(info.get("success",False)), "reason":str(info.get("termination_reason")), "constraints":";".join(info.get("violated_constraints",[]))}


def seed11_figures():
    model,vn,dummy,mp,vp=load_pair(11,600_000,600_000,"final")
    histories=[rollout(model,vn,s) for s in range(50)]; dummy.close()
    terminals=np.array([h["pos"][-1] for h in histories]); rep=int(np.argmin(abs(terminals-np.median(terminals))))
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.8),gridspec_kw={"width_ratios":[1.3,1]})
    for i,h in enumerate(histories):
        axs[0].plot(h["x"],h["y"],color=BLUE,lw=.75,alpha=.18)
        axs[1].plot(h["x"],h["y"],color=BLUE,lw=.75,alpha=.18)
    for ax in axs:
        h=histories[rep]; ax.add_patch(Circle((0,0),.1,facecolor=GREEN,edgecolor=GREEN,alpha=.22,lw=1.2,zorder=1)); ax.plot(h["x"],h["y"],color=DARK,lw=2.2,zorder=3)
        ax.grid(True,color="#D9D9D9",lw=.65,alpha=.75); ax.set_axisbelow(True); ax.invert_xaxis(); ax.set_xlabel(r"$r_{p,x}$ [m]",fontsize=12); ax.tick_params(axis="both",labelsize=11)
    # The true 0.1 m circle is sub-pixel at the full 300 m scale, so add a
    # small display-space marker at its centre to keep the tolerance visible.
    axs[0].scatter(0,0,s=42,facecolor=GREEN,edgecolor=GREEN,alpha=.45,lw=.8,zorder=4)
    axs[0].set_ylabel(r"$r_{p,y}$ [m]",fontsize=12); axs[0].set_title("Complete approach",loc="left",fontweight="normal")
    axs[1].set(xlim=(.35,-.12),ylim=(-.12,.12)); axs[1].set_title("Terminal region",loc="left",fontweight="normal")
    axs[1].set_aspect("equal",adjustable="box")
    fig.suptitle("Round 5, seed 11: docking-port-relative trajectories (50 rollouts)",fontweight="normal")
    fig.legend([Line2D([],[],color=BLUE,alpha=.35),Line2D([],[],color=DARK,lw=2.2),Patch(facecolor=GREEN,edgecolor=GREEN,alpha=.22)], ["Individual rollout",f"Median-terminal-distance rollout (seed {rep})",r"Position tolerance: $\|r_p\|\leq0.1$ m"],loc="lower center",ncol=3,frameon=False)
    fig.subplots_adjust(bottom=.20,wspace=.28); save(fig,"r5_s11_b600k_evaluation_trajectories")

    fields=[("pos",r"$\|r_p\|$ [m]",.1),("vel",r"$\|v_p\|$ [m/s]",.1),("att",r"$|\psi_e|$ [deg]",4.),("rate",r"$|\omega|$ [deg/s]",.2)]
    fig,axs=plt.subplots(2,2,figsize=(10,6.8),sharex=True)
    for ax,(key,label,limit) in zip(axs.flat,fields):
        for h in histories: ax.plot(h["time"],h[key],color=BLUE,lw=.65,alpha=.15)
        ax.plot(histories[rep]["time"],histories[rep][key],color=DARK,lw=2)
        ax.axhline(limit,color=RED,ls="--",lw=1.2,label="Docking requirement")
        ax.set_ylabel(label,fontsize=12); ax.tick_params(axis="both",labelsize=11); ax.grid(True,color="#D9D9D9",lw=.65,alpha=.75); ax.set_axisbelow(True)
    axs[1,0].set_xlabel("Time [s]",fontsize=12); axs[1,1].set_xlabel("Time [s]",fontsize=12)
    fig.suptitle("Round 5, seed 11: final-approach error convergence",fontweight="normal",y=.94)
    fig.legend([Line2D([],[],color=BLUE,alpha=.35),Line2D([],[],color=DARK,lw=2),Line2D([],[],color=RED,ls="--")],["Individual rollout",f"Representative seed {rep}","Terminal requirement"],loc="lower center",ncol=3,frameon=False)
    fig.subplots_adjust(bottom=.13,top=.86,hspace=.20,wspace=.26); save(fig,"r5_s11_b600k_state_convergence")
    np.savez_compressed(OUT/"r5_s11_b600k_trajectory_histories.npz", **{f"seed_{i}_{k}":v for i,h in enumerate(histories) for k,v in h.items() if isinstance(v,np.ndarray)})
    return {"round":5,"seed":11,"budget":600000,"buffer":600000,"checkpoint_step":600000,"checkpoint_type":"final","model":str(mp),"vecnormalize":str(vp),"n_rollouts":50,"success_rate":float(np.mean([h['success'] for h in histories])),"representative_seed":rep}


def seed19_figures():
    raw=RESULTS/"retained_policy_evaluations"/"reconstructed_supplemental_policies_50rollouts"/"all_rollouts.csv"
    d=pd.read_csv(raw); d=d[(d.training_seed==19)&(d.training_budget_steps==300000)].sort_values("eval_seed")
    fields=[("terminal_port_position_error_m",r"$\|r_p\|$ [m]",.1),("terminal_port_velocity_error_mps",r"$\|v_p\|$ [m/s]",.1),("terminal_attitude_error_deg",r"$|\psi_e|$ [deg]",4.),("terminal_angular_rate_error_deg_s",r"$|\omega|$ [deg/s]",.2)]
    fig,axs=plt.subplots(2,2,figsize=(10,6.8),sharex=True)
    for ax,(key,label,limit) in zip(axs.flat,fields):
        criterion=d[key]<=limit
        ax.scatter(d.loc[criterion,"eval_seed"],d.loc[criterion,key],s=24,color=GREEN,label="Criterion passed",zorder=3)
        ax.scatter(d.loc[~criterion,"eval_seed"],d.loc[~criterion,key],s=30,color=RED,marker="x",label="Criterion failed",zorder=4)
        failed=~d.success.astype(bool); ax.scatter(d.loc[failed,"eval_seed"],d.loc[failed,key],s=55,facecolors="none",edgecolors=DARK,lw=.9,label="Rollout failed (geometry)",zorder=5)
        ax.axhline(limit,color=RED,ls="--",lw=1.2); ax.set_ylabel(label); ax.grid(True,color="#D9D9D9",lw=.65,alpha=.75); ax.set_axisbelow(True)
    axs[1,0].set_xlabel("Evaluation seed"); axs[1,1].set_xlabel("Evaluation seed")
    fig.suptitle("Round 3 seed 19: absolute terminal errors versus docking requirements",fontweight="bold")
    handles=[Line2D([],[],marker="o",ls="",color=GREEN,label="Terminal criterion passed"),Line2D([],[],marker="x",ls="",color=RED,label="Terminal criterion failed"),Line2D([],[],marker="o",ls="",mfc="none",mec=DARK,label="Rollout failed: docking-cone geometry"),Line2D([],[],color=RED,ls="--",label="Requirement")]
    fig.legend(handles=handles,loc="lower center",ncol=4,frameon=False); fig.subplots_adjust(bottom=.14,hspace=.20)
    save(fig,"r3_s19_b300k_terminal_error_distributions")

    failed=d[~d.success.astype(bool)]; med=failed.terminal_port_position_error_m.median(); rep=int(failed.iloc[(failed.terminal_port_position_error_m-med).abs().argsort()[:1]].eval_seed.iloc[0])
    model,vn,dummy,mp,vp=load_pair(19,300_000,300_000,"best"); h=rollout(model,vn,rep); dummy.close()
    fig,axs=plt.subplots(1,3,figsize=(12.2,4.2))
    axs[0].plot(h["x"],h["y"],color=DARK,lw=1.6); axs[0].plot(h["x"],.1*h["x"],color=RED,ls="--",lw=1); axs[0].plot(h["x"],-.1*h["x"],color=RED,ls="--",lw=1); axs[0].scatter(0,0,s=120,color=GREEN,alpha=.3); axs[0].invert_xaxis(); axs[0].set(xlabel=r"$r_{p,x}$ [m]",ylabel=r"$r_{p,y}$ [m]",title="Docking-plane trajectory")
    axs[1].plot(h["time"],h["pos"],label=r"$\|r_p\|$",color=BLUE); axs[1].axhline(.1,color=RED,ls="--",label="0.1 m requirement"); axs[1].set(xlabel="Time [s]",ylabel="Position error [m]",title="Terminal position")
    margin=.1*h["s"]-h["rho"]; axs[2].plot(h["time"],margin,color=DARK); axs[2].axhline(0,color=RED,ls="--"); axs[2].set(xlabel="Time [s]",ylabel=r"Cone margin $0.1s-\rho$ [m]",title="Docking-cone geometry")
    for ax in axs: ax.grid(True,color="#D9D9D9",lw=.65,alpha=.75); ax.set_axisbelow(True)
    fig.suptitle(f"Round 3 seed 19: representative failed rollout (evaluation seed {rep})",fontweight="bold"); fig.subplots_adjust(wspace=.30)
    save(fig,"r3_s19_b300k_representative_failure")
    terminal_fail_counts={key:int((d[key]>limit).sum()) for key,_,limit in fields}
    reasons=Counter(d.loc[~d.success.astype(bool),"termination_reason"]); constraints=Counter(d.loc[~d.success.astype(bool),"violated_constraints"])
    return {"round":3,"seed":19,"budget":300000,"buffer":300000,"checkpoint_step":160000,"checkpoint_type":"best","model":str(mp),"vecnormalize":str(vp),"evaluation_file":str(raw),"n_rollouts":len(d),"success_rate":float(d.success.mean()),"failure_reasons":dict(reasons),"failure_constraints":dict(constraints),"terminal_criterion_failure_counts":terminal_fail_counts,"representative_failed_seed":rep,"representative_reason":h["reason"],"representative_constraints":h["constraints"]}


def main():
    style(); records=training_plots(); records.extend([seed11_figures(),seed19_figures()])
    with (OUT/"figure_source_summary.json").open("w",encoding="utf-8") as f: json.dump(records,f,indent=2)
    evaluation_csv = RESULTS / "retained_policy_evaluations" / "reconstructed_selected_policies_50rollouts" / "all_rollouts.csv"
    supplemental_csv = RESULTS / "retained_policy_evaluations" / "reconstructed_supplemental_policies_50rollouts" / "all_rollouts.csv"
    summary_rows = [
        {"round":5,"seed":3,"budget":600000,"buffer":600000,"checkpoint_step":100000,"checkpoint_type":"Intermediate checkpoint","files_used":f"learning_curve.csv; evaluations.npz; exact _best.zip; exact _best_vecnormalize.pkl; {evaluation_csv}","n_evaluation_rollouts":50,"success_rate_percent":100},
        {"round":4,"seed":10,"budget":600000,"buffer":100000,"checkpoint_step":570000,"checkpoint_type":"Intermediate checkpoint","files_used":f"learning_curve.csv; evaluations.npz; exact _best.zip; exact _best_vecnormalize.pkl; {evaluation_csv}","n_evaluation_rollouts":50,"success_rate_percent":100},
        {"round":5,"seed":11,"budget":600000,"buffer":600000,"checkpoint_step":600000,"checkpoint_type":"Final policy","files_used":"learning_curve.csv; evaluations.npz; exact final .zip; exact final _vecnormalize.pkl; newly collected r5_s11_b600k_trajectory_histories.npz","n_evaluation_rollouts":50,"success_rate_percent":100},
        {"round":3,"seed":19,"budget":300000,"buffer":300000,"checkpoint_step":160000,"checkpoint_type":"Intermediate checkpoint","files_used":f"{supplemental_csv}; exact _best.zip; exact _best_vecnormalize.pkl; evaluation seed 44 rerun for failure history","n_evaluation_rollouts":50,"success_rate_percent":42},
    ]
    with (OUT/"policy_summary.csv").open("w",newline="",encoding="utf-8") as f:
        keys=list(summary_rows[0]); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(summary_rows)
    print(f"Wrote figures and summaries to {OUT}")

if __name__ == "__main__": main()
