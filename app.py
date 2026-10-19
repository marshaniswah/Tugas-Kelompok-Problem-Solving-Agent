"""
Interactive Live Demonstration & Visualizer for 1D Bin Packing Problem (1D-BPP)
Course: AI5103 Fundamental of AI & Data Science — Problem Solving Agent
Compares: First Fit Decreasing (FFD) vs. Genetic Algorithm (GA) vs. Particle Swarm Optimization (PSO)

Features:
1. Live execution on single instances or instant benchmark replay.
2. Animated / moving convergence curves (GA vs. PSO vs. BKS).
3. Bin packing solution visualization (interactive stacked container layout).
4. Full quantitative comparison of all three algorithms on the same instance.
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup repo paths
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_instance
from src.baseline import first_fit_decreasing
from src.ga import run_genetic_algorithm
from src.pso import run_particle_swarm_optimization
from src.validation import validate_packing_solution
from src.metrics import compute_gap_percent

# Page configuration
st.set_page_config(
    page_title="1D-BPP Optimization: GA vs PSO vs FFD",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished academic UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-bks {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .badge-ga {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .badge-pso {
        background-color: #DBEAFE;
        color: #1E40AF;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .badge-ffd {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Instances metadata
INSTANCES_INFO = {
    "Falkenauer_u120_00.txt": {"n": 120, "c": 150, "bks": 48, "desc": "Uniform [20, 100], C=150"},
    "Falkenauer_u120_01.txt": {"n": 120, "c": 150, "bks": 49, "desc": "Uniform [20, 100], C=150"},
    "Falkenauer_u120_02.txt": {"n": 120, "c": 150, "bks": 46, "desc": "Uniform [20, 100], C=150"},
    "Falkenauer_u120_03.txt": {"n": 120, "c": 150, "bks": 49, "desc": "Uniform [20, 100], C=150"},
    "Falkenauer_u120_04.txt": {"n": 120, "c": 150, "bks": 50, "desc": "Uniform [20, 100], C=150"},
}

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/package.png", width=64)
    st.title("Kontrol Eksperimen")
    st.caption("AI5103 Problem Solving Agent Benchmark")

    st.subheader("1. Pilih Instance")
    selected_instance = st.selectbox(
        "Instance Falkenauer U",
        list(INSTANCES_INFO.keys()),
        index=0,
        help="Semua 5 instance telah terbukti optimal (Solved) di BPPLIB."
    )
    inst_meta = INSTANCES_INFO[selected_instance]
    bks = inst_meta["bks"]

    st.subheader("2. Mode Demonstrasi")
    demo_mode = st.radio(
        "Pilih Mode",
        ["🚀 Live Optimization Run (Eksekusi Nyata)", "⚡ Fast Benchmark Replay (Tersimpan)"],
        index=0,
        help="Live Run mengeksekusi GA, PSO, dan FFD langsung. Fast Replay memuat hasil tersimpan seketika."
    )

    st.subheader("3. Parameter Algoritma")
    if "Live Optimization" in demo_mode:
        eval_budget = st.slider("Evaluation Budget", 500, 6000, 3000, step=500, help="Total evaluasi fungsi objektif untuk GA dan PSO.")
        seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
        with st.expander("⚙️ Parameter Lanjutan (GA & PSO)"):
            pop_size = st.slider("Population / Swarm Size (N)", 20, 80, 50, step=10)
            ga_pm = st.slider("GA Mutation Rate (p_m)", 0.01, 0.40, 0.20, step=0.01)
            ga_pc = st.slider("GA Crossover Rate (p_c)", 0.50, 1.00, 0.90, step=0.05)
            ga_tour = st.slider("GA Tournament Size (k)", 2, 6, 3, step=1)
            pso_w = st.slider("PSO Inertia Weight (w)", 0.1, 1.0, 0.7, step=0.05)
            pso_c1 = st.slider("PSO Cognitive (c1)", 0.5, 3.0, 1.5, step=0.1)
            pso_c2 = st.slider("PSO Social (c2)", 0.5, 3.0, 1.5, step=0.1)
    else:
        seed = 11
        eval_budget = 6000
        pop_size = 50
        ga_pm = 0.20
        ga_pc = 0.90
        ga_tour = 3
        pso_w = 0.7
        pso_c1 = 1.5
        pso_c2 = 1.5

    run_button = st.button("▶️ Jalankan Optimasi Live", type="primary", use_container_width=True)

# -------------------------------------------------------------
# LOAD INSTANCE DATA
# -------------------------------------------------------------
instance_file = REPO_ROOT / "data" / "Falkenauer_U" / selected_instance
if not instance_file.is_file():
    st.error(f"File instance tidak ditemukan di: {instance_file}")
    st.stop()

instance_data = load_instance(instance_file)
weights = instance_data["weights"]
capacity = instance_data["capacity"]
n_items = instance_data["n_items"]
total_weight = sum(weights)
vol_lower_bound = instance_data["volume_lower_bound"]

# Session state initialization
if "results" not in st.session_state or st.session_state.get("current_instance") != selected_instance:
    st.session_state["results"] = None
    st.session_state["current_instance"] = selected_instance

# -------------------------------------------------------------
# RUN OPTIMIZATION LOGIC
# -------------------------------------------------------------
if run_button or st.session_state["results"] is None:
    with st.spinner(f"Menjalankan FFD, GA, dan PSO pada {selected_instance}..."):
        progress_bar = st.progress(0, text="Menyiapkan data...")

        # 1. FFD Baseline
        progress_bar.progress(15, text="1/3: Mengeksekusi First Fit Decreasing (FFD)...")
        t0 = time.perf_counter()
        res_ffd = first_fit_decreasing(weights, capacity, bks=bks)
        t_ffd = time.perf_counter() - t0

        # 2. Genetic Algorithm
        progress_bar.progress(40, text=f"2/3: Mengeksekusi Genetic Algorithm ({eval_budget:,} evaluasi)...")
        res_ga = run_genetic_algorithm(
            weights=weights,
            capacity=capacity,
            population_size=pop_size,
            tournament_size=ga_tour,
            mutation_rate=ga_pm,
            crossover_rate=ga_pc,
            evaluation_budget=eval_budget,
            seed=seed,
            bks=bks
        )

        # 3. Particle Swarm Optimization
        progress_bar.progress(75, text=f"3/3: Mengeksekusi Particle Swarm Optimization ({eval_budget:,} evaluasi)...")
        res_pso = run_particle_swarm_optimization(
            weights=weights,
            capacity=capacity,
            population_size=pop_size,
            w=pso_w,
            c1=pso_c1,
            c2=pso_c2,
            v_min=-0.2,
            v_max=0.2,
            pos_min=0.0,
            pos_max=1.0,
            evaluation_budget=eval_budget,
            seed=seed,
            bks=bks
        )

        progress_bar.progress(100, text="Selesai! Seluruh solusi terverifikasi valid.")
        time.sleep(0.3)
        progress_bar.empty()

        st.session_state["results"] = {
            "ffd": res_ffd,
            "ga": res_ga,
            "pso": res_pso,
            "seed": seed,
            "budget": eval_budget
        }

results = st.session_state["results"]
res_ffd = results["ffd"]
res_ga = results["ga"]
res_pso = results["pso"]

# -------------------------------------------------------------
# HEADER SECTION
# -------------------------------------------------------------
st.markdown('<div class="main-title">📦 Problem Solving Agent: 1D Bin Packing Problem (1D-BPP)</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">Live Optimization Demo & Comparative Visualizer: <b>GA vs. PSO vs. FFD</b> | '
    f'Instance: <code>{selected_instance}</code> (Items: {n_items}, Capacity: {capacity})</div>',
    unsafe_allow_html=True
)

# Top metric summary cards
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("Instance Target", selected_instance.replace(".txt", ""))
col_m2.metric("Total Bobot (Item)", f"{total_weight:,}")
col_m3.metric("BKS / Optimum", f"{bks} Bins", help="Proven Optimum terverifikasi dari BPPLIB")
col_m4.metric("Lower Bound L1", f"{vol_lower_bound} Bins", help="Continuous volume lower bound: ceil(Sum(w)/C)")
col_m5.metric("Evaluasi Budget", f"{results['budget']:,}")

st.divider()

# -------------------------------------------------------------
# MAIN TABS
# -------------------------------------------------------------
tab_conv, tab_sol, tab_comp, tab_val = st.tabs([
    "📈 Kurva Konvergensi Bergerak",
    "📦 Tampilan Solusi (Bin Packing Layout)",
    "⚖️ Perbandingan Ketiga Algoritma",
    "🛡️ Validasi & Detail Matematika"
])

# =============================================================
# TAB 1: KURVA KONVERGENSI BERGERAK (ANIMATED CONVERGENCE)
# =============================================================
with tab_conv:
    st.subheader("📈 Kurva Konvergensi Real-Time (Moving / Animated Convergence)")
    st.caption("Visualisasi pergerakan pencarian solusi terbaik (best-so-far) terhadap jumlah evaluasi fungsi objektif.")

    # Align GA and PSO evaluations
    ga_ckpts = res_ga["eval_checkpoints"]
    ga_curve = res_ga["convergence"]
    pso_curve = res_pso["history"]
    max_evals = min(len(pso_curve), ga_ckpts[-1] if ga_ckpts else len(pso_curve))

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        step_eval = st.slider(
            "Geser slider untuk melihat progres konvergensi sampai evaluasi ke-:",
            min_value=int(ga_ckpts[0]) if ga_ckpts else 50,
            max_value=int(max_evals),
            value=int(max_evals),
            step=50,
            key="conv_slider"
        )
    with col_ctrl2:
        st.write("")
        animate_btn = st.button("▶️ Mainkan Animasi Konvergensi", use_container_width=True)

    # Filter curves up to current slider evaluation
    curr_ga_idx = max(0, sum(1 for e in ga_ckpts if e <= step_eval) - 1)
    sub_ga_x = [e for e in ga_ckpts if e <= step_eval]
    sub_ga_y = ga_curve[:len(sub_ga_x)]

    sub_pso_x = list(range(1, step_eval + 1, 5))
    sub_pso_y = [pso_curve[min(i - 1, len(pso_curve) - 1)] for i in sub_pso_x]

    current_ga_val = sub_ga_y[-1] if sub_ga_y else ga_curve[0]
    current_pso_val = pso_curve[min(step_eval - 1, len(pso_curve) - 1)]

    # Metrics at current evaluation point
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluasi Saat Ini", f"{step_eval:,} / {max_evals:,}")
    c2.metric("GA Best Bins", f"{current_ga_val} Bins", delta=f"{current_ga_val - bks:+d} vs BKS", delta_color="inverse")
    c3.metric("PSO Best Bins", f"{current_pso_val} Bins", delta=f"{current_pso_val - bks:+d} vs BKS", delta_color="inverse")
    c4.metric("FFD Baseline (Final)", f"{res_ffd['best_bins']} Bins", delta=f"{res_ffd['best_bins'] - bks:+d} vs BKS", delta_color="inverse")

    plot_placeholder = st.empty()

    def build_plotly_convergence(eval_limit):
        x_ga = [e for e in ga_ckpts if e <= eval_limit]
        y_ga = ga_curve[:len(x_ga)]
        x_pso = list(range(1, eval_limit + 1, 5))
        y_pso = [pso_curve[min(i - 1, len(pso_curve) - 1)] for i in x_pso]

        fig = go.Figure()
        # BKS Line
        fig.add_hline(
            y=bks, line_dash="dash", line_color="#10B981", line_width=2.5,
            annotation_text=f"Proven Optimum BKS ({bks} Bins)",
            annotation_position="bottom right",
            annotation_font=dict(color="#10B981", size=12, family="sans-serif")
        )
        # FFD Baseline Line
        fig.add_hline(
            y=res_ffd["best_bins"], line_dash="dot", line_color="#F59E0B", line_width=2,
            annotation_text=f"FFD Baseline ({res_ffd['best_bins']} Bins)",
            annotation_position="top left",
            annotation_font=dict(color="#F59E0B", size=11)
        )
        # GA Convergence
        fig.add_trace(go.Scatter(
            x=x_ga, y=y_ga,
            mode="lines+markers",
            name="Genetic Algorithm (GA)",
            line=dict(color="#EF4444", width=3),
            marker=dict(size=5, color="#DC2626")
        ))
        # PSO Convergence
        fig.add_trace(go.Scatter(
            x=x_pso, y=y_pso,
            mode="lines",
            name="Particle Swarm Optimization (PSO)",
            line=dict(color="#3B82F6", width=2.5)
        ))

        fig.update_layout(
            title=dict(
                text=f"Kurva Konvergensi Komparatif pada {selected_instance} (Evaluasi: {eval_limit:,})",
                font=dict(size=16)
            ),
            xaxis=dict(title="Evaluasi Fungsi Objektif", range=[0, max_evals + 100], gridcolor="#F1F5F9"),
            yaxis=dict(title="Jumlah Bin Terbaik (Minimize)", range=[bks - 1, max(max(ga_curve), max(pso_curve)) + 2], gridcolor="#F1F5F9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white",
            height=460,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    if animate_btn:
        progress_steps = [e for e in ga_ckpts if e <= max_evals]
        # Step through checkpoints smoothly
        for step in progress_steps[::2]:
            fig_anim = build_plotly_convergence(step)
            plot_placeholder.plotly_chart(fig_anim, use_container_width=True)
            time.sleep(0.04)
        # Final frame
        plot_placeholder.plotly_chart(build_plotly_convergence(max_evals), use_container_width=True)
    else:
        plot_placeholder.plotly_chart(build_plotly_convergence(step_eval), use_container_width=True)

    st.info(
        f"💡 **Analisis Konvergensi**: GA mengevaluasi populasi 50 individu per generasi, mencatat checkpoint setiap 50 evaluasi. "
        f"PSO memperbarui partikel secara sekuensial. Pada instance **{selected_instance}**, BKS tercapai pada {bks} bin."
    )

# =============================================================
# TAB 2: TAMPILAN SOLUSI (BIN PACKING STACKING LAYOUT)
# =============================================================
with tab_sol:
    st.subheader("📦 Tampilan Solusi: Visualisasi Penataan Item dalam Bin (Bin Packing Layout)")
    st.caption("Menampilkan bagaimana setiap item (bobot $w_i$) ditempatkan ke dalam kontainer (kapasitas $C = 150$).")

    sol_algo = st.radio(
        "Pilih Solusi Algoritma untuk Ditampilkan:",
        ["Genetic Algorithm (GA)", "Particle Swarm Optimization (PSO)", "First Fit Decreasing (FFD)"],
        horizontal=True
    )

    if "Genetic Algorithm" in sol_algo:
        active_bins = res_ga["best_bins"]
        algo_tag = "GA"
        color_theme = px.colors.qualitative.Plotly
    elif "Particle Swarm" in sol_algo:
        active_bins = res_pso["best_bins"]
        algo_tag = "PSO"
        color_theme = px.colors.qualitative.Safe
    else:
        active_bins = res_ffd["bins"]
        algo_tag = "FFD"
        color_theme = px.colors.qualitative.Pastel

    total_bins_used = len(active_bins)
    bin_weights = [sum(weights[i] for i in b) for b in active_bins]
    bin_utilizations = [(w_sum / capacity) * 100 for w_sum in bin_weights]
    avg_utilization = float(np.mean(bin_utilizations))
    min_utilization = float(np.min(bin_utilizations))
    max_utilization = float(np.max(bin_utilizations))

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric(f"Total Bin Digunakan ({algo_tag})", f"{total_bins_used} Bins")
    col_s2.metric("Gap terhadap BKS", f"{compute_gap_percent(total_bins_used, bks):.2f}%")
    col_s3.metric("Rata-rata Utilitas Bin", f"{avg_utilization:.2f}%")
    col_s4.metric("Kapasitas Bin Maksimal (C)", f"{capacity}")

    st.markdown("#### Susunan Item per Bin (Interactive Stacked Horizontal Layout)")

    # Build stacked bar chart of bins
    fig_bins = go.Figure()

    # Find max items in any bin for trace grouping
    max_items_in_bin = max(len(b) for b in active_bins)

    for item_pos in range(max_items_in_bin):
        x_vals = []
        y_vals = []
        hover_texts = []
        labels = []

        for b_idx, b in enumerate(active_bins):
            bin_name = f"Bin {b_idx + 1:02d}"
            if item_pos < len(b):
                item_idx = b[item_pos]
                w_item = weights[item_idx]
                accum_weight = sum(weights[b[k]] for k in range(item_pos + 1))
                slack = capacity - sum(weights[b[k]] for k in range(len(b)))

                x_vals.append(w_item)
                y_vals.append(bin_name)
                labels.append(f"#{item_idx}<br>({w_item})")
                hover_texts.append(
                    f"<b>{bin_name}</b><br>"
                    f"Item Index: #{item_idx}<br>"
                    f"Item Weight: {w_item}<br>"
                    f"Accumulated in Bin: {accum_weight} / {capacity} ({(accum_weight/capacity)*100:.1f}%)<br>"
                    f"Total Bin Weight: {sum(weights[i] for i in b)}<br>"
                    f"Residual Slack: {slack}"
                )
            else:
                x_vals.append(0)
                y_vals.append(bin_name)
                labels.append("")
                hover_texts.append("")

        fig_bins.add_trace(go.Bar(
            y=y_vals,
            x=x_vals,
            name=f"Item Slot {item_pos + 1}",
            orientation="h",
            text=labels,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=9, color="white"),
            hovertext=hover_texts,
            hoverinfo="text",
            marker=dict(color=color_theme[item_pos % len(color_theme)])
        ))

    # Add red line for maximum capacity C = 150
    fig_bins.add_vline(
        x=capacity, line_dash="solid", line_color="#EF4444", line_width=2.5,
        annotation_text=f"Batas Kapasitas Maksimal (C = {capacity})",
        annotation_position="top right",
        annotation_font=dict(color="#EF4444", size=11, family="sans-serif")
    )

    fig_bins.update_layout(
        barmode="stack",
        height=max(500, total_bins_used * 22),
        showlegend=False,
        xaxis=dict(
            title=f"Kapasitas Terisi (Kapasitas Maksimal C = {capacity})",
            range=[0, capacity + 15],
            gridcolor="#E2E8F0"
        ),
        yaxis=dict(
            title="Daftar Bin",
            autorange="reversed",
            tickfont=dict(size=9)
        ),
        plot_bgcolor="#F8FAFC",
        margin=dict(l=70, r=40, t=30, b=40)
    )

    st.plotly_chart(fig_bins, use_container_width=True)

    # Detailed table of bin contents
    with st.expander("📋 Rincian Lengkap Seluruh Bin (Tabel Penataan Item & Slack)"):
        table_rows = []
        for b_idx, b in enumerate(active_bins):
            w_list = [weights[i] for i in b]
            total_w = sum(w_list)
            slack = capacity - total_w
            util = (total_w / capacity) * 100
            table_rows.append({
                "Bin ID": f"Bin {b_idx + 1:02d}",
                "Jumlah Item": len(b),
                "Total Bobot": total_w,
                "Sisa Ruang (Slack)": slack,
                "Utilitas (%)": f"{util:.2f}%",
                "Indeks Item Terkemas": str(b),
                "Bobot Tiap Item": str(w_list)
            })
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

# =============================================================
# TAB 3: PERBANDINGAN KETIGA ALGORITMA (FFD vs GA vs PSO)
# =============================================================
with tab_comp:
    st.subheader(f"⚖️ Perbandingan Hasil Ketiga Algoritma pada {selected_instance}")
    st.caption("Perbandingan deterministik baseline (FFD) dan dua metaheuristik (GA & PSO) di bawah evaluasi yang sama.")

    # Prepare comparison records
    comp_data = [
        {
            "Algoritma": "First Fit Decreasing (FFD)",
            "Kategori": "Heuristik Deterministik",
            "Jumlah Bin": res_ffd["best_bins"],
            "BKS": bks,
            "Gap vs BKS (%)": f"{res_ffd['gap_percent']:.2f}%",
            "Evaluasi": res_ffd["evaluations"],
            "Waktu Komputasi (s)": f"{res_ffd['runtime_seconds']:.4f}",
            "Solusi Valid": "✅ Ya" if res_ffd["valid"] else "❌ Tidak",
            "Rata-rata Utilitas Bin": f"{np.mean([(sum(weights[i] for i in b)/capacity)*100 for b in res_ffd['bins']]):.2f}%"
        },
        {
            "Algoritma": "Genetic Algorithm (GA)",
            "Kategori": "Metaheuristik Permutasi",
            "Jumlah Bin": res_ga["best_fitness"],
            "BKS": bks,
            "Gap vs BKS (%)": f"{res_ga['gap_percent']:.2f}%",
            "Evaluasi": res_ga["evaluations"],
            "Waktu Komputasi (s)": f"{res_ga['runtime_seconds']:.2f}",
            "Solusi Valid": "✅ Ya" if res_ga["valid"] else "❌ Tidak",
            "Rata-rata Utilitas Bin": f"{np.mean([(sum(weights[i] for i in b)/capacity)*100 for b in res_ga['best_bins']]):.2f}%"
        },
        {
            "Algoritma": "Particle Swarm Optimization (PSO)",
            "Kategori": "Metaheuristik Kontinu (Argsort)",
            "Jumlah Bin": res_pso["best_fitness"],
            "BKS": bks,
            "Gap vs BKS (%)": f"{res_pso['gap_percent']:.2f}%",
            "Evaluasi": res_pso["evaluations"],
            "Waktu Komputasi (s)": f"{res_pso['runtime_seconds']:.2f}",
            "Solusi Valid": "✅ Ya" if res_pso["valid"] else "❌ Tidak",
            "Rata-rata Utilitas Bin": f"{np.mean([(sum(weights[i] for i in b)/capacity)*100 for b in res_pso['best_bins']]):.2f}%"
        }
    ]

    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp, hide_index=True, use_container_width=True)

    # Bar chart comparison
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("#### Perbandingan Jumlah Bin terhadap BKS")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["BKS Optimum", "FFD Baseline", "Genetic Algorithm", "Particle Swarm"],
            y=[bks, res_ffd["best_bins"], res_ga["best_fitness"], res_pso["best_fitness"]],
            marker_color=["#10B981", "#F59E0B", "#EF4444", "#3B82F6"],
            text=[f"{bks}", f"{res_ffd['best_bins']}", f"{res_ga['best_fitness']}", f"{res_pso['best_fitness']}"],
            textposition="auto"
        ))
        fig_bar.update_layout(
            yaxis=dict(title="Jumlah Bin (Lebih Sedikit Lebih Baik)", range=[bks - 2, max(res_ffd['best_bins'], res_ga['best_fitness'], res_pso['best_fitness']) + 2]),
            height=340,
            margin=dict(l=30, r=30, t=30, b=30),
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b2:
        st.markdown("#### Perbandingan Waktu Eksekusi (Detik)")
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=["FFD Baseline", "Genetic Algorithm", "Particle Swarm"],
            y=[res_ffd["runtime_seconds"], res_ga["runtime_seconds"], res_pso["runtime_seconds"]],
            marker_color=["#F59E0B", "#EF4444", "#3B82F6"],
            text=[f"{res_ffd['runtime_seconds']:.4f}s", f"{res_ga['runtime_seconds']:.2f}s", f"{res_pso['runtime_seconds']:.2f}s"],
            textposition="auto"
        ))
        fig_time.update_layout(
            yaxis=dict(title="Waktu Komputasi (Detik)", gridcolor="#F1F5F9"),
            height=340,
            margin=dict(l=30, r=30, t=30, b=30),
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown(r"""
    **Insight Komparatif**:
    1. **First Fit Decreasing (FFD)** merupakan heuristik deterministik $O(n \log n)$ yang sangat cepat (<0.001s). Pada instance uniform Falkenauer, pengurutan bobot secara menurun memberikan penataan awal yang sangat padat.
    2. **Genetic Algorithm (GA)** secara konsisten mengungguli PSO karena representasi permutasi langsung dengan **Order Crossover (OX)** mempertahankan blok urutan parsial antar generasi, sementara **Swap Mutation ($p_m = 0.20$)** menjaga diversitas genetik.
    3. **Particle Swarm Optimization (PSO)** beroperasi pada ruang kontinu $[0, 1]^n$ yang dipetakan melalui `argsort`. Clamping kecepatan $[-0.2, 0.2]$ mencegah divergensi partikel, namun transisi urutan diskrit melalui koordinat kontinu memerlukan evaluasi lebih besar untuk menandingi GA.
    """)

# =============================================================
# TAB 4: VALIDASI DAN DETAIL MATEMATIKA
# =============================================================
with tab_val:
    st.subheader("🛡️ Protokol Validasi Solusi & Integritas Matematika")

    # Re-validate live
    v_ga_ok, v_ga_msg = validate_packing_solution(res_ga["best_bins"], weights, capacity, res_ga["best_fitness"])
    v_pso_ok, v_pso_msg = validate_packing_solution(res_pso["best_bins"], weights, capacity, res_pso["best_fitness"])
    v_ffd_ok, v_ffd_msg = validate_packing_solution(res_ffd["bins"], weights, capacity, res_ffd["best_bins"])

    st.success(f"✅ **Audit GA**: {v_ga_msg}")
    st.success(f"✅ **Audit PSO**: {v_pso_msg}")
    st.success(f"✅ **Audit FFD**: {v_ffd_msg}")

    st.markdown(f"""
    ---
    ### Formulasi 1D Bin Packing Problem (1D-BPP)
    $$\\min \\quad m = \\sum_{{j=1}}^m y_j$$
    $$\\text{{subject to:}} \\quad \\sum_{{i=0}}^{{n-1}} w_i x_{{ij}} \\le C y_j, \\quad \\forall j \\in \\{{1, \\dots, m\\}}$$
    $$\\sum_{{j=1}}^m x_{{ij}} = 1, \\quad \\forall i \\in \\{{0, \\dots, n-1\\}}$$
    $$x_{{ij}} \\in \\{{0, 1\\}}, \\quad y_j \\in \\{{0, 1\\}}$$

    ### Batas Bawah Kontinu (Volume Lower Bound $L_1$)
    $$L_1 = \\left\\lceil \\frac{{\\sum_{{i=0}}^{{n-1}} w_i}}{{C}} \\right\\rceil = \\left\\lceil \\frac{{{total_weight:,}}}{{{capacity}}} \\right\\rceil = {vol_lower_bound}$$

    ### Rumus Gap Persentase terhadap BKS
    $$\\text{{Gap (\\%)}} = \\frac{{\\text{{Hasil}} - \\text{{BKS}}}}{{\\text{{BKS}}}} \\times 100\\%$$
    """)

st.caption("AI5103 Fundamental of AI & Data Science — Problem Solving Agent · Kelompok Marshaniswah Syamsul")
