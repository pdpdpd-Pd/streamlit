import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import re
from collections import defaultdict
import sys
import os

# 设置页面配置
st.set_page_config(page_title="Motter-Lai 动态恢复分析", layout="wide")

# 设置绘图风格和字体（支持中文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {"font.sans-serif": ['SimHei']})

# 尝试导入原始类
try:
    from Analysis import DynamicRecoveryAnalysis
except ImportError:
    st.error("无法导入 Analysis.py。请确保 Analysis.py 与当前脚本在同一目录下。")
    st.stop()


class StreamlitDynamicAnalysis(DynamicRecoveryAnalysis):
    """
    专门适配 Streamlit 的分析类。
    继承自 Analysis.py 中的 DynamicRecoveryAnalysis。
    修改了绘图方法以返回 Figure 对象而不是直接 plt.show()。
    """

    def get_binary_phase_fig(self, target_net_type, target_centrality, metric):
        """生成二元相图 Figure"""
        data_storage = {1: defaultdict(list), 2: defaultdict(list)}
        found_data = False

        for data_key, k_data in self.data.items():
            parts = data_key.split('_')
            if len(parts) < 2 or parts[0] != target_net_type or parts[1] != target_centrality:
                continue

            param_key = parts[-1]
            alpha_match = re.search(r'A(\d+)', param_key)
            recovery_match = re.search(r'R(\d+)', param_key)

            if not alpha_match or not recovery_match:
                continue

            alpha = int(alpha_match.group(1))
            R = int(recovery_match.group(1))

            for k_val in [1, 2]:
                if k_val in k_data:
                    found_data = True
                    df = k_data[k_val]
                    try:
                        if 'origin' in df['phase'].values:
                            origin_df = df[df['phase'] == 'origin'].iloc[0]
                        else:
                            origin_df = df.iloc[0]

                        origin_eff = origin_df['network_efficiency']
                        origin_nodes = origin_df['max_component_size'] if 'max_component_size' in df.columns else 200

                        cascade_df = df[df['phase'] == 'cascade']
                        steady_df = cascade_df.iloc[-1:] if len(cascade_df) > 0 else df.iloc[-1:]

                        if len(steady_df) == 0: continue

                        if metric == 'efficiency_ratio':
                            val = steady_df['network_efficiency'].iloc[0] / origin_eff if origin_eff > 1e-6 else 0
                        elif metric == 'nodes_ratio':
                            failed = steady_df['failed_nodes_count'].iloc[0] if 'failed_nodes_count' in steady_df else 0
                            val = (origin_nodes - failed) / origin_nodes
                        else:
                            val = 0

                        if val >= 0:
                            data_storage[k_val][(alpha, R)].append(val)
                    except Exception:
                        pass

        if not found_data:
            return None

        plot_dfs = {}
        all_values = []
        for k_val in [1, 2]:
            rows = []
            for (alpha, R), values in data_storage[k_val].items():
                if values:
                    mean_val = np.mean(values)
                    rows.append({'alpha': alpha, 'R': R, 'value': mean_val})
                    all_values.append(mean_val)
            plot_dfs[k_val] = pd.DataFrame(rows)

        if not all_values: return None

        vmin, vmax = min(all_values), max(all_values)
        if vmax - vmin < 1e-6: vmin -= 0.1; vmax += 0.1

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        plt.subplots_adjust(wspace=0.1, right=0.85)
        cmap = plt.cm.jet

        im_objects = []
        for idx, k_val in enumerate([1, 2]):
            ax = axes[idx]
            df = plot_dfs[k_val]
            if df.empty:
                ax.text(0.5, 0.5, "无数据", ha='center', va='center')
                im_objects.append(None)
                continue

            alpha_min, alpha_max = df['alpha'].min(), df['alpha'].max()
            R_min, R_max = df['R'].min(), df['R'].max()

            # 只有单个点时的保护
            if alpha_min == alpha_max: alpha_max += 0.5; alpha_min -= 0.5
            if R_min == R_max: R_max += 0.5; R_min -= 0.5

            grid_x, grid_y = np.mgrid[alpha_min:alpha_max:200j, R_min:R_max:200j]

            try:
                grid_z = griddata((df['alpha'], df['R']), df['value'], (grid_x, grid_y), method='cubic')
            except:
                grid_z = griddata((df['alpha'], df['R']), df['value'], (grid_x, grid_y), method='linear')

            im = ax.imshow(grid_z.T, extent=(alpha_min, alpha_max, R_min, R_max),
                           origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            im_objects.append(im)

            ax.set_title(f'k = {k_val}', fontsize=12)
            ax.set_xlabel(r'参数 $\alpha$')
            if idx == 0: ax.set_ylabel('恢复数量 R')
            ax.grid(False)

        valid_im = next((im for im in im_objects if im is not None), None)
        if valid_im:
            cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(valid_im, cax=cbar_ax)
            cbar.set_label("效率比率" if metric == 'efficiency_ratio' else "节点留存率")

        fig.suptitle(f'{target_net_type} - {target_centrality} - {metric} 二元相图', y=0.98)
        return fig

    def get_grouped_distribution_fig(self, target_net_type, target_centrality, metric):
        """生成分组分布图 (Seaborn catplot)"""
        all_records = []
        for data_key, k_data in self.data.items():
            parts = data_key.split('_')
            if len(parts) < 2 or parts[0] != target_net_type or parts[1] != target_centrality: continue

            param_key = parts[-1]
            alpha_match = re.search(r'A(\d+)', param_key)
            recovery_match = re.search(r'R(\d+)', param_key)
            if not alpha_match or not recovery_match: continue

            alpha = int(alpha_match.group(1))
            R = int(recovery_match.group(1))

            for k_val, df in k_data.items():
                try:
                    origin_eff = df[df['phase'] == 'origin'].iloc[0]['network_efficiency'] if 'origin' in df[
                        'phase'].values else df.iloc[0]['network_efficiency']
                    steady_row = df[df['phase'] == 'cascade'].iloc[-1] if not df[df['phase'] == 'cascade'].empty else \
                    df.iloc[-1]

                    if metric == 'efficiency_ratio':
                        val = steady_row['network_efficiency'] / origin_eff if origin_eff > 0 else 0
                    elif metric == 'nodes_ratio':
                        origin_nodes = 200
                        rem = origin_nodes - steady_row.get('failed_nodes_count', 0)
                        val = rem / origin_nodes
                    else:
                        val = 0

                    all_records.append({'Alpha': alpha, 'R': R, 'k': f'k={k_val}', 'Value': val})
                except:
                    pass

        if not all_records: return None

        df_plot = pd.DataFrame(all_records)
        g = sns.catplot(
            data=df_plot, x='Alpha', y='Value', hue='R', col='k',
            kind='violin', palette='viridis', height=5, aspect=1.2,
            inner='quartile', cut=0
        )
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle(f'{target_net_type} - {target_centrality} 参数分布 ({metric})', fontsize=16)
        return g.fig

    def get_jittered_heatmap_fig(self, target_net_type, target_centrality, metric):
        """生成抖动散点热力图"""
        df = self._extract_all_data_to_df(target_net_type, target_centrality, metric)
        if df.empty: return None

        jitter_strength = 0.3
        df['Alpha_Jitter'] = df['Alpha'] + np.random.uniform(-jitter_strength, jitter_strength, len(df))
        df['R_Jitter'] = df['R'] + np.random.uniform(-jitter_strength, jitter_strength, len(df))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        cmap = plt.cm.viridis

        for idx, k_val in enumerate([1, 2]):
            ax = axes[idx]
            subset = df[df['k'] == k_val]
            if subset.empty: continue

            sc = ax.scatter(
                subset['Alpha_Jitter'], subset['R_Jitter'],
                c=subset['Metric'], cmap=cmap, s=30, alpha=0.6,
                edgecolors='white', linewidth=0.2
            )
            ax.set_xticks(sorted(df['Alpha'].unique()))
            ax.set_yticks(sorted(df['R'].unique()))
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_title(f'k={k_val} 原始数据点')
            ax.set_xlabel('Alpha')
            if idx == 0: ax.set_ylabel('Recovery R')

            plt.colorbar(sc, ax=ax, label=metric)

        fig.suptitle(f'{target_net_type} - {target_centrality} 全数据抖动视图', fontsize=16)
        return fig

    def get_gain_phase_fig(self, target_net_type, target_centrality, metric):
        """生成增益相图 (k2 - k1)"""
        X, Y, Z1, Z2 = self._get_phase_grid_data(target_net_type, target_centrality, metric)
        if Z1 is None or Z2 is None: return None

        Gain = Z2 - Z1
        fig, ax = plt.subplots(figsize=(10, 8))
        extent = [X.min(), X.max(), Y.min(), Y.max()]

        im = ax.imshow(Gain.T, extent=extent, origin='lower', aspect='auto', cmap='inferno')

        try:
            max_gain = np.nanmax(Gain)
            if max_gain > 0.05:
                levels = np.linspace(0.05, max_gain, 5)
                contours = ax.contour(X, Y, Gain, levels=levels, colors='cyan', linewidths=0.8)
                ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        except:
            pass

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'效能增益 ({metric})')
        ax.set_title(f'{target_net_type} - {target_centrality}: k=2 相比 k=1 增益')
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Recovery R')
        return fig


# --- Streamlit 界面逻辑 ---

# 1. 侧边栏：数据加载
st.sidebar.header("📂 数据配置")
default_path = "C:\\Users\\李芃荻\\PycharmProjects\\master\\masterK\\motter-lai恢复"
data_path = st.sidebar.text_input("CSV结果文件夹路径:", value=default_path)


# 修改1：兼容旧版本的缓存装饰器
@st.cache(allow_output_mutation=True)
def load_analysis_data(path):
    """缓存数据加载，避免重复读取"""
    if not os.path.exists(path):
        return None
    analyzer = StreamlitDynamicAnalysis(path)
    return analyzer


if st.sidebar.button("重新加载数据") or 'analyzer' not in st.session_state:
    with st.spinner("正在扫描并加载数据，请稍候..."):
        loaded_analyzer = load_analysis_data(data_path)
        if loaded_analyzer:
            st.session_state['analyzer'] = loaded_analyzer
            st.sidebar.success(f"成功加载数据！\n共 {len(loaded_analyzer.data)} 组")
        else:
            st.sidebar.error("路径不存在，请检查输入。")

# 2. 主界面：分析控制
st.title("📊 Motter-Lai 模型恢复策略分析")

if 'analyzer' in st.session_state:
    analyzer = st.session_state['analyzer']

    # 创建两列布局
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("🛠️ 参数选择")

        net_types = ["WS", "BA", "CM", "ER"]
        selected_net = st.selectbox("网络类型 (Network Type)", net_types, index=0)

        centralities = ["bc", "cc"]
        selected_cent = st.selectbox("中心性指标 (Centrality)", centralities, index=0)

        metrics = {"Efficiency Ratio": "efficiency_ratio", "Nodes Ratio": "nodes_ratio"}
        selected_metric_name = st.selectbox("分析指标 (Metric)", list(metrics.keys()), index=1)
        selected_metric = metrics[selected_metric_name]

        st.write("---")
        st.subheader("📈 图表类型")
        plot_type = st.radio(
            "选择要绘制的图表:",
            ("二元相图 (Binary Phase)",
             "分组分布图 (Box/Violin)",
             "抖动热力图 (Jittered Heatmap)",
             "增益相图 (Gain Phase)")
        )

        # 修改2：兼容旧版本的按钮参数（去掉了 type 和 use_container_width）
        run_btn = st.button("开始绘图")

    with col2:
        if run_btn:
            st.subheader(f"分析结果: {selected_net} - {selected_cent}")

            if plot_type == "二元相图 (Binary Phase)":
                with st.spinner("正在生成平滑相图..."):
                    fig = analyzer.get_binary_phase_fig(selected_net, selected_cent, selected_metric)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.warning("没有找到足够的数据来生成相图。")

            elif plot_type == "分组分布图 (Box/Violin)":
                with st.spinner("正在统计分布数据..."):
                    fig = analyzer.get_grouped_distribution_fig(selected_net, selected_cent, selected_metric)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.warning("数据不足。")

            elif plot_type == "抖动热力图 (Jittered Heatmap)":
                with st.spinner("正在绘制所有数据点..."):
                    fig = analyzer.get_jittered_heatmap_fig(selected_net, selected_cent, selected_metric)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.warning("数据不足。")

            elif plot_type == "增益相图 (Gain Phase)":
                with st.spinner("正在计算 k=2 增益..."):
                    fig = analyzer.get_gain_phase_fig(selected_net, selected_cent, selected_metric)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.warning("无法生成增益图（可能缺少 k=1 或 k=2 的配对数据）。")
        else:
            st.info("请在左侧选择参数并点击“开始绘图”")

else:
    st.info("👈 请先在侧边栏确认数据路径并点击“加载数据”")
