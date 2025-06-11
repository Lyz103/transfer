# plot
# -*- coding: utf-8 -*-
# @Author : LuyuChen
# @Time : 2024/10/30 16:52

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# plt.rc('font', family='Calibri')
import seaborn
import numpy as np
import json
import matplotlib
import os
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import make_interp_spline

# matplotlib.use('TkAgg')

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_compare_result(file_name, if_raw, fig_name):
    ####### 雷达图
    # 属性列表
    attributes = ["Naturalness", "Clarity", "Adaptability", "Relevance", "Roleplay", "Realism", "Overall"]
    num_attributes = len(attributes)

    # 读取并处理数据
    if if_raw:
        with open(file_name, "r", encoding="utf-8") as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
    else:
        with open(file_name, "r", encoding="utf-8") as f:
            dat = json.load(f)
        data = [dat[key] for key in dat.keys()]

    for i in range(len(data)):
        data[i] = data[i]["compare_result"]

    # 统计各个属性的 win、draw、loss 概率
    count = {}
    for attr in attributes:
        count[attr] = {"Win": 0, "Draw": 0, "Loss": 0}
    for d in data:
        for attr in attributes:
            if d[attr.lower()] == 1:
                count[attr]["Win"] += 1
            elif d[attr.lower()] == 0:
                count[attr]["Draw"] += 1
            else:
                count[attr]["Loss"] += 1

    # 计算各个属性的 win、draw、loss 概率
    total_evaluations = len(data)
    win_ratio = [count[attr]["Win"] / total_evaluations * 100 for attr in attributes]
    draw_ratio = [count[attr]["Draw"] / total_evaluations * 100 for attr in attributes]
    loss_ratio = [count[attr]["Loss"] / total_evaluations * 100 for attr in attributes]

    # 为雷达图准备角度和数据
    angles = np.linspace(0, 2 * np.pi, num_attributes, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 计算旋转角度，使“Overall”位于顶部（π/2）
    rotation = (np.pi / 2) - angles[-2]  # angles[-1]是“Overall”的原始角度

    # 应用旋转，使“Overall”位于顶部
    angles = [(angle + rotation) % (2 * np.pi) for angle in angles]

    win_ratio += win_ratio[:1]
    draw_ratio += draw_ratio[:1]
    loss_ratio += loss_ratio[:1]

    # 调整全局字体大小和样式
    plt.rcParams.update({'font.size': 18})
    # 定义颜色
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']  # 与之前的样式匹配

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(16, 8), subplot_kw=dict(polar=True))  # 增加图的大小

    # 绘制 win, draw, loss 的数据线和填充
    data_per_outcome = [("Win", win_ratio, colors[0]), ("Draw", draw_ratio, colors[1]), ("Loss", loss_ratio, colors[2])]

    for outcome_name, outcome_values, color in data_per_outcome:
        ax.plot(angles, outcome_values, label=outcome_name, color=color, linewidth=2)
        ax.fill(angles, outcome_values, alpha=0.25, color=color)

    # 设置属性标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes, fontsize=34, ha='center')

    # 调整属性标签位置，使其更美观
    for label, angle in zip(ax.get_xticklabels(), angles):
        x, y = label.get_position()
        lab_angle = np.rad2deg(angle)
        if lab_angle > 90 and lab_angle < 270:
            label.set_horizontalalignment('right')
        elif lab_angle == 270 or lab_angle == 90:
            label.set_horizontalalignment('center')
        else:
            label.set_horizontalalignment('left')
        label.set_position((x, y - 0.008))  # 向外移动标签

    # 设置径向刻度标签
    ax.set_rlabel_position(89)
    rmax = 100  # 直接设定最大值为100
    radial_ticks = np.arange(0, rmax + 10, 20)  # 从0到100，步长为5
    # radial_tick_labels = [f"" for tick in radial_ticks[:-2]] + [f"{radial_ticks[-2]}%"] + [f""]
    radial_tick_labels = [f"" for tick in radial_ticks[:-2]] + [f""] + [f""]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(radial_tick_labels, color="grey", size=28)
    ax.set_ylim(0, rmax)

    # 调整径向标签位置
    ax.tick_params(axis='y', pad=10)

    # 添加标题和图例
    # 从文件名中提取模型名称
    print(file_name)
    model_name_1 = file_name.split("/")[-2].split("_")[0]
    model_name_2 = file_name.split("/")[-1].split("_")[2].split(".")[0]
    if "OurUser" in model_name_1:
        model_name_1 = model_name_1.replace("OurUser", "Ours")
    else:
        model_name_1 = model_name_1.replace("User", "")

    if "-9b-chat" in model_name_1:
        model_name_1 = model_name_1.replace("-9b-chat", "")
    if "gpt-4o-mini" in model_name_1:
        model_name_1 = model_name_1.replace("gpt-4o-mini", "4o-mini")
    if "gpt-4o-mini" in model_name_2:
        model_name_2 = model_name_2.replace("gpt-4o-mini", "4o-mini")

    if "OurUser" in model_name_2:
        model_name_2 = model_name_2.replace("OurUser", "Ours")
    else:
        model_name_2 = model_name_2.replace("User", "")

    ax.set_title(f'{model_name_1} vs {model_name_2}', y=1.15, fontsize=45)  # 调整标题位置

    # 调整图例位置
    ax.legend(loc='upper right', bbox_to_anchor=(1.8, 1.1), fontsize=28)

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()


def plot_result(output_paths: list, max_turn=5, info_amount_bound=[2, 4], text_length_bound=[16, 40]):
    # 读取所有文件的数据
    all_data = {}
    for output_path in output_paths:
        with open(os.path.join(output_path, f"result_{max_turn}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        model_name = output_path.split("/")[-1].split("_")[0]
        all_data[model_name] = {}
        plt.style.use('ggplot')  # 更美观的风格
        all_data[model_name]["Total LengthJudger"] = data["Total LengthJudger"]
        all_data[model_name]["Total InfoAmountJudger"] = data["Total InfoAmountJudger"]
        all_data[model_name]["Total FormalityJudger"] = data["Total FormalityJudger"]

    # 比较分析部分
    # 1、绘制各个模型的总长度分布对比
    plt.figure()
    for model_name, data in all_data.items():
        length_judger = data["Total LengthJudger"]
        x = sorted([int(k) for k in length_judger.keys()])
        y = [length_judger.get(str(i), 0) / sum(length_judger.values()) * 100 for i in x]  # 计算占比
        # 画出分布图,使用小点连接,实线
        plt.plot(x, y, label=model_name, marker=',', linestyle='-', alpha=0.6)
    plt.xticks(range(0, max(x) + 1, 40))  # 设置刻度间隔
    plt.title("Length Distribution Comparison")
    plt.xlabel("Length")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.show()

    # 2、绘制信息量对比
    plt.figure()
    for model_name, data in all_data.items():
        info_amount_judger = data["Total InfoAmountJudger"]
        x = sorted([int(k) for k in info_amount_judger.keys()])
        y = [info_amount_judger.get(str(i), 0) / sum(info_amount_judger.values()) * 100 for i in x]  # 计算占比
        plt.plot(x, y, label=model_name, marker=',', linestyle='-', alpha=0.6)
    plt.title("Information Amount Distribution Comparison")
    plt.xlabel("Info Amount")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.show()

    # 3、正式程度占比比较图（非正式和正式分开展示）
    formality_data = {}
    for model_name, data in all_data.items():
        formality_judger = data["Total FormalityJudger"]
        total_count = sum(formality_judger.values())
        formality_data[model_name] = [formality_judger.get(str(i), 0) / total_count * 100 for i in [0, 1]]  # 非正式和正式

    labels = ["Informal", "Formal"]
    x = np.arange(len(labels))
    width = 0.35  # 柱状图宽度
    fig, ax = plt.subplots()
    for i, model_name in enumerate(all_data.keys()):
        ax.bar(x - width / 2 + i * width, formality_data[model_name], width, label=model_name, alpha=0.7)
    ax.set_xlabel("Formality")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Formality Distribution Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

    # 4、长度和信息量的短、中、长与低、中、高分布对比
    # plt.figure()
    length_data = {}
    for model_name, data in all_data.items():
        length_judger = data["Total LengthJudger"]
        length_data[model_name] = [0, 0, 0]
        for k, v in length_judger.items():
            if int(k) < text_length_bound[0]:
                length_data[model_name][0] += v
            elif text_length_bound[0] <= int(k) < text_length_bound[1]:
                length_data[model_name][1] += v
            else:
                length_data[model_name][2] += v
        # 求百分比
        total = sum(length_judger.values())
        length_data[model_name] = [v / total * 100 for v in length_data[model_name]]

    labels = ["Short", "Medium", "Long"]
    x = np.arange(len(labels))
    width = 0.35  # 柱状图宽度
    fig, ax = plt.subplots()
    for i, model_name in enumerate(all_data.keys()):
        ax.bar(x - width / 2 + i * width, length_data[model_name], width, label=model_name, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Length")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Length Range Comparison")
    ax.legend()
    plt.show()

    # 5、Info Amount: Low, Medium, High
    # plt.figure()
    info_amount_data = {}
    for model_name, data in all_data.items():
        info_amount_judger = data["Total InfoAmountJudger"]
        info_amount_data[model_name] = [0, 0, 0]
        for k, v in info_amount_judger.items():
            if int(k) < info_amount_bound[0]:
                info_amount_data[model_name][0] += v
            elif info_amount_bound[0] <= int(k) < info_amount_bound[1]:
                info_amount_data[model_name][1] += v
            else:
                info_amount_data[model_name][2] += v
        # 求百分比
        total = sum(info_amount_judger.values())
        info_amount_data[model_name] = [v / total * 100 for v in info_amount_data[model_name]]

    labels = ["Low", "Medium", "High"]
    x = np.arange(len(labels))
    width = 0.35  # 柱状图宽度
    fig, ax = plt.subplots()
    for i, model_name in enumerate(all_data.keys()):
        ax.bar(x - width / 2 + i * width, info_amount_data[model_name], width, label=model_name, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Info Amount")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Info Amount Range Comparison")
    ax.legend()
    plt.show()


def plot_class_result(output_paths_class: list, max_turn=5, info_amount_bound=[2, 3], text_length_bound=[12, 24],
                      class_names: list = ["UserSim", "iEvaLM", "CSHI"]):
    # Increase global font size for readability
    # plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 18})

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Define colors for the models

    all_data = {}
    for i, output_paths in enumerate(output_paths_class):
        all_data[class_names[i]] = {}
        all_data[class_names[i]]["Total LengthJudger"] = {}
        all_data[class_names[i]]["Total InfoAmountJudger"] = {}
        all_data[class_names[i]]["Total FormalityJudger"] = {}
        for output_path in output_paths:
            with open(os.path.join(output_path, f"result_{max_turn}.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            now_length_judger = data["Total LengthJudger"]
            now_info_amount_judger = data["Total InfoAmountJudger"]
            now_formality_judger = data["Total FormalityJudger"]

            # Aggregate data
            for judger_name, judger_data in [("Total LengthJudger", now_length_judger),
                                             ("Total InfoAmountJudger", now_info_amount_judger),
                                             ("Total FormalityJudger", now_formality_judger)]:
                for k, v in judger_data.items():
                    if k not in all_data[class_names[i]][judger_name]:
                        all_data[class_names[i]][judger_name][k] = 0
                    all_data[class_names[i]][judger_name][k] += v

    # Function to create radar charts
    def plot_radar_chart(categories, data_per_model, title, colors):

        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        # Close the plot by completing the loop
        angles += angles[:1]

        # Find the maximum percentage to set the radial axis limit
        max_percentage = max([max(values) for _, values in data_per_model])
        rmax = (np.ceil(max_percentage / 10) * 10) if max_percentage > 0 else 100  # Round up to the nearest 10

        # Initialize radar plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, (model_name, values) in enumerate(data_per_model):
            values += values[:1]  # Complete the loop for each model's data

            ax.plot(angles, values, label=model_name, color=colors[i % len(colors)], linewidth=2)
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        # Set category labels with increased font size
        ax.set_xticks(angles[:-1])
        # Adjust position of category labels
        ax.set_xticklabels(categories, fontsize=20, ha='center')

        # Move category labels outward
        for label, angle in zip(ax.get_xticklabels(), angles):
            x, y = label.get_position()
            lab_angle = np.rad2deg(angle)
            if lab_angle >= 90 and lab_angle <= 270:
                label.set_horizontalalignment('right')
            else:
                label.set_horizontalalignment('left')
            label.set_position((x, y - 0.1))  # Move label outward

        # Set radial labels with smaller font size
        ax.set_rlabel_position(0)
        # Define radial ticks based on rmax
        radial_ticks = np.linspace(0, rmax, num=5)
        radial_tick_labels = [f"{int(tick)}%" for tick in radial_ticks[:-1]] + [""]
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels(radial_tick_labels, color="grey", size=12)  # Smaller font size for radial labels
        ax.set_ylim(0, rmax)

        # Adjust position to prevent overlap
        ax.tick_params(axis='y', pad=10)  # Increase padding for radial labels

        # Add title and legend with increased font size
        ax.set_title(title, y=1.08, fontsize=24)
        # Position the legend outside the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.15), fontsize=16)
        plt.tight_layout()
        plt.savefig(f"Judge{categories}.png")
        plt.show()

    # Prepare data for each radar chart
    variables_list = ["Length", "Info Amount", "Formality"]
    categories_list = [
        ["Short", "Medium", "Long"],  # Length categories
        ["Low", "Medium", "High"],  # Info Amount categories
        ["", "Informal", "Formal"]  # Formality categories with a dummy category
    ]

    # Plot radar charts for Length, Info Amount, and Formality
    for idx, variable in enumerate(variables_list):
        data_per_model = []
        categories = categories_list[idx]

        for model_name in class_names:
            judger_key = f"Total {variable.replace(' ', '')}Judger"
            judger_data = all_data[model_name][judger_key]

            # Handle length and information amount bounds
            if variable == "Length":
                bounds = text_length_bound
                counts = [0, 0, 0]
                for k, v in judger_data.items():
                    if int(k) < bounds[0]:
                        counts[0] += v
                    elif bounds[0] <= int(k) < bounds[1]:
                        counts[1] += v
                    else:
                        counts[2] += v
            elif variable == "Info Amount":
                bounds = info_amount_bound
                counts = [0, 0, 0]
                for k, v in judger_data.items():
                    if int(k) < bounds[0]:
                        counts[0] += v
                    elif bounds[0] <= int(k) < bounds[1]:
                        counts[1] += v
                    else:
                        counts[2] += v
            elif variable == "Formality":
                # For Formality, we have only two categories: '0' (Informal) and '1' (Formal)
                counts = [0, judger_data.get("0", 0), judger_data.get("1", 0)]  # Adding a dummy zero count

            total = sum(counts)
            percentages = [count / total * 100 if total > 0 else 0 for count in counts]
            data_per_model.append((model_name, percentages))

        # Plot the radar chart for this variable
        plot_radar_chart(categories, data_per_model, f"{variable} Distribution Comparison", colors)


def plot_controlability(result_file_name, history_file_name,
                        info_amount_bound=[2, 4],
                        text_length_bound=[16, 40]):
    plt.style.use('ggplot')  # 更美观的风格
    # 增加全局字体大小
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['axes.titlesize'] = 19.5
    plt.rcParams['axes.labelsize'] = 15.5
    plt.rcParams['xtick.labelsize'] = 13.5
    plt.rcParams['ytick.labelsize'] = 13.5
    plt.rcParams['legend.fontsize'] = 13.5
    model_name = result_file_name.split("/")[-2].split("_")[0].replace("OurUser", "")

    # keys
    Length_keys = ["简短", "中等", "冗长"]
    Length_keys2en = {"简短": "Short Message", "中等": "Medium Message", "冗长": "Long Message"}
    Length_values = ["Short", "Medium", "Long"]
    InfoAmount_keys = ["信息量少", "信息量适中", "信息量大"]
    InfoAmount_keys2en = {"信息量少": "Uninformative", "信息量适中": "Medium Informative", "信息量大": "Informative"}
    InfoAmount_values = ["Low", "Medium", "High"]
    Formality_keys = ["正式", "非正式"]
    Formality_keys2en = {"正式": "Formal\nSpeech", "非正式": "Informal\nSpeech "}
    Formality_values = ["Informal", "Formal"]

    Length_counter = {k: {v: 0 for v in Length_values} for k in Length_keys}
    InfoAmount_counter = {k: {v: 0 for v in InfoAmount_values} for k in InfoAmount_keys}
    Formality_counter = {k: {v: 0 for v in Formality_values} for k in Formality_keys}

    Length_register = {k: {} for k in Length_keys}
    InfoAmount_register = {k: {} for k in InfoAmount_keys}
    Formality_register = {k: {} for k in Formality_keys}

    # 读取文件
    with open(history_file_name, "r", encoding="utf-8") as f:
        data = f.readlines()
    history_data = [json.loads(d) for d in data]

    with open(result_file_name, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    # 遍历所有历史，统计"信息量少"、"信息量适中"、"信息量大"
    # "user_profile" -> "user_behavior" -> "words_chara" -> "信息量"/"正式与否"/"语句长度"
    # "id" 对应到result_data中的key
    t = 0

    for history in history_data:
        id = history["id"]
        # print(eval(history["user_profile"]))
        user_behavior = eval(history["user_profile"])['user_behavior']
        # print(type(user_behavior))
        # 信息量
        info_amount = user_behavior["words_chara"]["信息量"][0]
        # 正式与否
        formality = user_behavior["words_chara"]["正式与否"][0]
        # 语句长度
        length = user_behavior["words_chara"]["语句长度"][0]
        #  or (info_amount == "信息量少" and length == "中等")
        if (info_amount == "信息量少" and length == "冗长") or (info_amount == "信息量大" and length == "简短") or (
                info_amount == "信息量少" and length == "中等"):
            # 不合理的组合
            # print(history)
            continue        
        # 统计
        try:
            count_result = result_data[str(id)]
            t += 1
            # "LengthJudger", "InfoAmountJudger", "FormalityJudger"
            for keys in count_result["LengthJudger"]:
                if int(keys) < text_length_bound[0]:
                    Length_counter[length]["Short"] += count_result["LengthJudger"][keys]
                elif text_length_bound[0] <= int(keys) < text_length_bound[1]:
                    Length_counter[length]["Medium"] += count_result["LengthJudger"][keys]
                else:
                    Length_counter[length]["Long"] += count_result["LengthJudger"][keys]

                if keys not in Length_register[length]:
                    Length_register[length][keys] = 0
                Length_register[length][keys] += count_result["LengthJudger"][keys]
        except:
            pass
        try:
            for keys in count_result["InfoAmountJudger"]:
                if int(keys) < info_amount_bound[0]:
                    InfoAmount_counter[info_amount]["Low"] += count_result["InfoAmountJudger"][keys]
                elif info_amount_bound[0] <= int(keys) < info_amount_bound[1]:
                    InfoAmount_counter[info_amount]["Medium"] += count_result["InfoAmountJudger"][keys]
                else:
                    InfoAmount_counter[info_amount]["High"] += count_result["InfoAmountJudger"][keys]

                if keys not in InfoAmount_register[info_amount]:
                    InfoAmount_register[info_amount][keys] = 0
                InfoAmount_register[info_amount][keys] += count_result["InfoAmountJudger"][keys]

                # if int(keys) == 4: print()
                # print(1)
        except:
            pass
        try:
            for keys in count_result["FormalityJudger"]:
                if int(keys) == 0:
                    Formality_counter[formality]["Informal"] += count_result["FormalityJudger"][keys]
                else:
                    Formality_counter[formality]["Formal"] += count_result["FormalityJudger"][keys]

                if keys not in Formality_register[formality]:
                    Formality_register[formality][keys] = 0

        except:
            pass
    print(t)
    color_1 = (31 / 255, 119 / 255, 180 / 255)
    color_2 = (255 / 255, 127 / 255, 14 / 255)
    color_3 = (44 / 255, 160 / 255, 44 / 255)
    colors = [color_1, color_2, color_3]
    # # 绘制图表
    # 1、绘制长度分布,
    fig, ax = plt.subplots()

    for key in Length_keys:
        # 获取数据并归一化
        x = np.array(sorted([int(k) for k in Length_register[key].keys()]))
        # # 删除大于100的异常值
        x = x[x <= 100]
        y = np.array([Length_register[key][str(i)] for i in x])
        y = y / sum(y) * 100

        # 插值拟合
        spline = make_interp_spline(x, y)
        x_smooth = np.linspace(min(x), max(x), 500)
        y_smooth = spline(x_smooth)
        # 进一步平滑曲线，减小震荡
        y_smooth = gaussian_filter1d(y_smooth, sigma=40)  # sigma 可以调整平滑度
        # 确保 y 值非负
        y_smooth = np.maximum(y_smooth, 0)
        # 绘制平滑曲线
        ax.plot(x_smooth, y_smooth, label=Length_keys2en[key], alpha=0.7, color=colors[Length_keys.index(key)%3])
        # 绘制阴影区域
        ax.fill_between(x_smooth, y_smooth, color=ax.get_lines()[-1].get_color(), alpha=0.3)
        # 标注原始数据点
        ax.scatter(x, y, color=ax.get_lines()[-1].get_color(), s=5, alpha=0.8)
        # # 颜色打印
        # print(ax.get_lines()[-1].get_color())


    ax.set_xlabel("Length")
    ax.set_ylabel("Proportion (%)")
    ax.set_title(f"Length Distribution{model_name}")
    # x,y轴大于0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    plt.savefig("Length.png")
    plt.show()

    # 2、绘制信息量分布
    # 绘制具体的分布,拟合出分布曲线，同上
    fig, ax = plt.subplots()

    for key in InfoAmount_keys:
        # 获取数据并归一化
        x = np.array(sorted([int(k) for k in InfoAmount_register[key].keys()]))
        y = np.array([InfoAmount_register[key][str(i)] for i in x])
        y = y / sum(y) * 100

        # 差值拟合
        spline = make_interp_spline(x, y)
        x_smooth = np.linspace(min(x), max(x), 500)
        y_smooth = spline(x_smooth)
        # 进一步平滑曲线，减小震荡
        y_smooth = gaussian_filter1d(y_smooth, sigma=40)
        # 确保 y 值非负
        y_smooth = np.maximum(y_smooth, 0)
        # 绘制平滑曲线
        ax.plot(x_smooth, y_smooth, label=InfoAmount_keys2en[key], alpha=0.7, color=colors[InfoAmount_keys.index(key)%3])
        # 绘制阴影区域
        ax.fill_between(x_smooth, y_smooth, color=ax.get_lines()[-1].get_color(), alpha=0.3)
        # 标注原始数据点
        ax.scatter(x, y, color=ax.get_lines()[-1].get_color(), s=5, alpha=0.8)

    ax.set_xlabel("Info Amount")
    ax.set_ylabel("Proportion (%)")
    ax.set_title(f"Info Amount Distribution{model_name}")

    # x,y轴大于0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    plt.savefig("Info.png")
    plt.show()

    # 3、绘制正式程度分布
    # 定义新的颜色组合
    colors = ['#1f78b4', '#33a02c']  # 蓝色和绿色

    # 计算各类别的正式和非正式占比
    formal_proportions = []
    informal_proportions = []
    for key in Formality_keys:
        total = sum(Formality_counter[key].values())
        formal = Formality_counter[key].get('Formal', 0) / total * 100
        informal = Formality_counter[key].get('Informal', 0) / total * 100
        formal_proportions.append(formal)
        informal_proportions.append(informal)

    # 定义图表和轴
    fig, ax = plt.subplots(figsize=(6.3, 2.7))

    # 设置柱形图的参数
    height = 0.05  # 每个柱形的高度
    ylim = 0.22

    # 手动设置 y 坐标以控制间距
    y_positions = [0.15, 0.05]  # 第一个柱形的中心在 0.4，第二个在 0.2

    # 绘制堆积水平条形图
    ax.barh(y_positions, informal_proportions, height, label='Informal', color=colors[0], alpha=0.8)
    ax.barh(y_positions, formal_proportions, height, left=informal_proportions, label='Formal', color=colors[1],
            alpha=0.8)

    # 设置 y 轴刻度和标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels([Formality_keys2en[key] for key in Formality_keys])

    # 设置轴标签和标题
    ax.set_xlabel("Proportion (%)")
    # ax.set_ylabel("Categories", fontsize=12)
    ax.set_title(f"Formality Distribution {model_name}")

    # 调整 y 轴范围以控制顶部和底部的空白
    ax.set_ylim(0, ylim)  # y 轴从 0 到 0.6

    # 将图例放在图表外部，避免与柱形重叠
    ax.legend()

    # 添加网格线
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()
    plt.savefig("Formality.png")
    plt.show()


if __name__ == "__main__":
    # 论文图二
    path = os.path.abspath(os.path.dirname(__file__))
    # plot_compare_result(os.path.join(path, "outputs/OurUser_gpt-4o-mini_gpt-4o-mini_TTS_BaseCRSgpt-4o-mini/compare_result_RecUsergpt-4o-mini_5_raw.jsonl"),if_raw=True, fig_name="a.png")
    # plot_compare_result(os.path.join(path, "outputs/OurUser_gpt-4o-mini_gpt-4o-mini_TTS_BaseCRSgpt-4o-mini/compare_result_OurUser_gpt-4o-mini_gpt-4o-mini_Prob_Sample_5_raw.jsonl"),if_raw=True, fig_name="b.png")
    # plot_compare_result(os.path.join(path, "outputs/OurUser_gpt-4o-mini_gpt-4o-mini_TTS_BaseCRSgpt-4o-mini/compare_result_OurUser_gpt-4o-mini_gpt-4o-mini_Uniform_Sample_5_raw.jsonl"),if_raw=True, fig_name="e.png")
    # plot_compare_result(os.path.join(path, "outputs/OurUser_gpt-4o-mini_gpt-4o-mini_Uniform_Sample_BaseCRSgpt-4o-mini/compare_result_RecUsergpt-4o-mini_5_raw.jsonl"),if_raw=True, fig_name="f.png")
    plot_compare_result(os.path.join(path, "/data/liyuanzi/HUAWEI/GUsim_V3/src/test_Beauty/OurUser_gpt-4o-mini_gpt-4o-mini_Prob_Sample_BaseCRSgpt-4o-mini/compare_result_RecUserBeautygpt-4o-mini_5_raw.jsonl"),if_raw=True, fig_name="test.png")
    # plot_compare_result(os.path.join(path, "outputs//OurUser(glm-4-9b-chat)_BaseCRS(gpt-4o-mini)//compare_result_iEvaLMUser(gpt-4o)_5_raw.jsonl"),if_raw=True)
    # plot_compare_result(os.path.join(path, "outputs//OurUser(gpt-4o-mini)_BaseCRS(gpt-4o-mini)//compare_result_iEvaLMUser(gpt-4o)_5_raw.jsonl"),if_raw=True)

    # 论文图三
    # path = os.path.abspath(os.path.dirname(__file__))

    # usersim_file_names = [os.path.join(path, "outputs//OurUser(gpt-4o-mini)_BaseCRS(gpt-4o-mini)"),
    #                       os.path.join(path, "outputs//OurUser(glm-4-9b-chat)_BaseCRS(gpt-4o-mini)"),
    #                       os.path.join(path, "outputs//OurUser(gpt-3.5-turbo)_BaseCRS(gpt-4o-mini)"),
    #                       ]
    # ievalm_file_names = [os.path.join(path, "outputs//iEvaLMUser(gpt-4o-mini)_BaseCRS(gpt-4o-mini)"),
    #                      ]
    # ievalm_file_names = [os.path.join(path, "outputs//iEvaLMUsergpt-4o-mini_BaseCRSgpt-4o-mini"),
    #                      ]
    # cshi_file_names = [os.path.join(path, "outputs//CSHIUser(gpt-4o-mini)_BaseCRS(gpt-4o-mini)"),
    #                    os.path.join(path, "outputs//CSHIUser(gpt-3.5-turbo)_BaseCRS(gpt-4o-mini)"),
    #                    os.path.join(path, "outputs//CSHIUser(gpt-4o)_BaseCRS(gpt-4o-mini)"),
    #                    ]    
    # cshi_file_names = [os.path.join(path, "outputs//CSHIUsergpt-4o-mini_BaseCRSgpt-4o-mini"),

    #                    ]
    # plot_class_result([usersim_file_names, ievalm_file_names, cshi_file_names])

    # recusersim_file_names = [os.path.join(path, "outputs//RecUsergpt-4o-mini_BaseCRSgpt-4o-mini"),
    #                    ]
    # omini_file_names = [os.path.join(path, "outputs/OurUsergpt-4o-minigpt-4o-mini_BaseCRSgpt-4o-mini"),

    #                    ]
    # usersim_file_names = [os.path.join(path, "outputs/OurUsergpt-4o-minideepseek-reasoner_BaseCRSgpt-4o-mini"),
    #                       ]
    # plot_class_result([omini_file_names, recusersim_file_names], class_names=["4o-mini", "RecUserSim"])
    # plot_class_result([usersim_file_names, omini_file_names, recusersim_file_names], class_names=["DS", "4o-mini", "RecUserSim"])

    # # # 论文图四
    # path = os.path.abspath(os.path.dirname(__file__))
    # plot_controlability(os.path.join(path, "outputs/OurUsergpt-4o-minideepseek-reasoner_BaseCRSgpt-4o-mini/result_5.json"),
    #                     os.path.join(path, "outputs/OurUsergpt-4o-minideepseek-reasoner_BaseCRSgpt-4o-mini/history_5.jsonl"))

    # plot_controlability(os.path.join(path, "outputs//OurUser(glm-4-9b-chat)_BaseCRS(gpt-4o-mini)//result_5.json"),
    #                     os.path.join(path, "outputs//OurUser(glm-4-9b-chat)_BaseCRS(gpt-4o-mini)//history_5.jsonl"))

    # plot_controlability(os.path.join(path, "outputs//OurUser(gpt-3.5-turbo)_BaseCRS(gpt-4o-mini)//result_5.json"),
    #                     os.path.join(path, "outputs//OurUser(gpt-3.5-turbo)_BaseCRS(gpt-4o-mini)//history_5.jsonl"))
