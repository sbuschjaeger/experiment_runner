#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import json

from os import listdir
from os.path import isfile, join

from itertools import cycle

import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def str_to_latex(original):
	# TODO implement Latex mapping. Currently not part of streamlit though?
	return original.replace("{","\\{").replace("}","\\}").replace("_", " ")

def row_to_nice_name(row):
	nice_name = row["model"] + " " + row["base_estimator"] + " with "

	if "regularizer" in row and row["regularizer"] == "None":
		nice_name += "No regularization "
	if "regularizer" in row and not row["regularizer"] == "None":
		nice_name += str(row["regularizer"]).replace("lambda","l") + " "
	
	# if "n_estimators" in row and not row["n_estimators"] == "None":
	# 	nice_name += str(row["n_estimators"]) +  " estimators "

	return nice_name

# def rows(df, key, col):
# 	dff = df[df["nice_name"] == key]
# 	return dff.groupby(["K"])[col].mean(), dff.groupby(["K"])[col].std()

def make_lines(df, method_name, color, show_legend, visible, x_axis, y_axis):
	dff = df[df["nice_name"] == method_name]

	if dff[x_axis].values[0] == "None":
		x_df = df[x_axis]
		x_df = x_df[x_df != "None"]
		x = np.unique(x_df)
		if y_axis + "_std" in list(dff):
			y_mean, y_std = dff[y_axis].mean(), dff[y_axis + "_std"].mean()
		else:
			y_mean, y_std = dff[y_axis].mean(), dff[y_axis].std()

		y_mean = [y_mean for _ in x]
		y_std = [y_std for _ in x]
	else:
		x = np.unique(dff[x_axis])
		if y_axis + "_std" in list(dff):
			y_mean, y_std = dff.groupby([x_axis])[y_axis].mean(), dff.groupby([x_axis])[y_axis + "_std"].mean()
		else:
			y_mean, y_std = dff.groupby([x_axis])[y_axis].mean(), dff.groupby([x_axis])[y_axis].std()
		
	l_name = str_to_latex(method_name)
	
	return go.Scatter(x=x, y=y_mean, line=dict(color=color), name=l_name, showlegend = show_legend,
					  visible = visible, 
					  error_y = dict(type="data", array=y_std, visible=not bool(np.any(np.isnan(y_std))))
					 )

def read_data(path):
	df = pd.read_json(path, lines=True)
	df = df.fillna("None")
	df["nice_name"] = df.apply(row_to_nice_name, axis=1)
	
	return df

@st.cache(allow_output_mutation=True)
def read_files(base_path):
	fnames = [f for f in listdir(base_path) if isfile(join(base_path, f)) and f.endswith("jsonl")]
	dfs = {}
	for fname in fnames:
		print("Reading " + join(base_path, fname))
		dfs[fname] = read_data(join(base_path, fname))

	return dfs

def plot_selected(df, selected_configs, metrics):
	if len(metrics) == 1:
		cols = rows = 1
	else:
		cols = 2
		rows = int(len(metrics) / cols) 
		if len(metrics) % cols != 0:
			rows += 1

	found_metrics = []
	for m in metrics:
		if m in list(df):
			found_metrics.append(m)

	fig = make_subplots(rows=rows,cols=cols,subplot_titles=found_metrics, vertical_spacing=0.1)

	if len(selected_configs) == 0:
		for r in range(1,rows+1):
			for c in range(1,cols+1):
				fig.append_trace(go.Scatter(x=[],y=[]),row = r,col = c)
	else:
		for method,color in zip(selected_configs, cycle(colors)):
			visible = True
			r = 1
			c = 1

			for m in found_metrics:
				first = (c == 1 and r == 1)
				fig.append_trace(make_lines(df, method, color, first, visible, "n_estimators", m),row=r,col=c)
				c += 1
				if c > 2:
					c = 1
					r += 1

	r = 1
	c = 1
	for m in found_metrics:
		fig.update_xaxes(title_text="K", row=r, col=c)
		fig.update_yaxes(title_text=m, row=r, col=c)
		c += 1
		if c > 2:
			c = 1
			r += 1

	return fig

if len(sys.argv) > 1:
	base_path = sys.argv[1]
else:
	print("ERROR: Folder path needed")
	sys.exit()

st.title('Deep Learning Ensemble Experiments')

# st.markdown(
# 		f"""
# <style>
# 	.reportview-container .main .block-container{{
# 		max-width: 1000px;
# 		padding-top: 1rem;
# 		padding-right: 1rem;
# 		padding-left: 1rem;
# 		padding-bottom: 1rem;
# 	}}
# 	.reportview-container .main {{
# 		color: black;
# 		background-color: white;
# 	}}
# </style>
# """,
# 		unsafe_allow_html=True
# 	)

#print("Reading files")
dfs = read_files(base_path)
#print("Files read.")
selected_df = st.selectbox('Select results file to display',list(dfs.keys()))

if isfile(base_path + "/" + selected_df.split(".csv")[0] + "-settings.json"):
	with open(base_path + "/" + selected_df.split(".csv")[0] + "-settings.json") as json_file:
		data = json.load(json_file)
	default_configs = data["selected_configs"]
	default_comments = data["comments"] 
	default_show_raw = data["show_raw"]
	default_show_legend = data["show_legend"]
else:
	default_configs = []
	default_comments = ""
	default_show_raw = False
	default_show_legend = False

df = dfs[selected_df]

st.subheader("Loaded " + selected_df)

colors = [
	'#1F77B4', # muted blue
	'#FF7F0E',  # safety orange
	'#2CA02C',  # cooked asparagus green
	'#D62728',  # brick red
	'#9467BD',  # muted purple
	'#8C564B',  # chestnut brown
	'#E377C2',  # raspberry yogurt pink
	'#7F7F7F',  # middle gray
	'#BCBD22',  # curry yellow-green
	'#17BECF'  # blue-teal
]

st.sidebar.subheader("Select configurations to plot")

show_raw = st.checkbox('Show raw entries', value=default_show_raw)
if show_raw:
	st.subheader('Raw results')
	st.write(df)

all_configs = np.unique(df["nice_name"]).tolist()
selected_configs = []
for cfg_name,color in zip(all_configs, cycle(colors)):
	agree = st.sidebar.checkbox(cfg_name, value=cfg_name in default_configs)
	if agree:
		selected_configs.append(cfg_name)

plot_metrics = ["accuracy_test", "accuracy_train", "complexity", "fit_time"]

fig = go.Figure(plot_selected(df, selected_configs, plot_metrics))

show_legend = st.sidebar.checkbox('Show legend entries', value=default_show_legend)

st.subheader("Comments")
comments = st.text_area("Comments", value=default_comments)

fig.update_layout(height=900, showlegend=show_legend, legend=dict(x=0,y=-0.2), legend_orientation="h")

st.plotly_chart(fig, height=1500 )

store_config = st.button("Store config")
if store_config:
	json_dict = {}
	json_dict["selected_configs"] = selected_configs
	json_dict["comments"] = comments
	json_dict["show_raw"] = show_raw
	json_dict["show_legend"] = show_legend

	with open(base_path + "/" + selected_df.split(".csv")[0] + "-settings.json" , 'w') as outfile:
		json.dump(json_dict, outfile)
	st.text("Config stored in " + base_path + "/" + selected_df.split(".csv")[0] + "-settings.json")

save_to_pdf = st.button("Store plots as PDF")
if save_to_pdf:
	pgf_template = """
\\documentclass[tikz,border=5pt]{standalone}

\\usepackage{xcolor}
\\usepackage{pgfplots}
\\pgfplotsset{compat=newest}
{colors}

\\begin{document}
\\begin{tikzpicture}
	\\begin{axis}[
		xlabel={xlabel},
		ylabel={ylabel},
		legend pos=north west
	]

{plots}

	\\end{axis}
\\end{tikzpicture}
\\end{document}
	"""
	pgf_path = base_path + "/" + selected_df.split(".csv")[0] 
	for m in plot_metrics:
		pgf_path_metric = pgf_path + "_" + m + ".tex"
		x_axis = "K"
		y_axis = m
		plot_str = ""

		for method,color in zip(selected_configs, cycle(colors)): 
			dff = df[df["nice_name"] == method]
			x = np.unique(dff[x_axis])
			if y_axis + "_std" in list(dff):
				y_mean, y_std = dff.groupby([x_axis])[y_axis].mean(), dff.groupby([x_axis])[y_axis + "_std"].mean()
			else:
				y_mean, y_std = dff.groupby([x_axis])[y_axis].mean(), dff.groupby([x_axis])[y_axis].std()

			coord = "\n\t\t\t".join(["({}, {})".format(xi,yi) for xi, yi in zip(x,y_mean)])
			plot_str += "\t\t\\addplot[mark=none,color={color}] coordinates { {coord} }; \n\t\t\\addlegendentry{{legend_name}} \n".replace("{coord}", coord).replace("{legend_name}", method).replace("{color}", color.replace("#",""))

		color_str = "\n".join(["\\definecolor{{value}}{HTML}{{value}}".replace("{value}", color.replace("#","")) for color in colors])
		pgf_str = pgf_template.replace("{colors}", color_str).replace("{xlabel}", x_axis).replace("{ylabel}", y_axis).replace("{plots}", plot_str)
		with open(pgf_path_metric, "w") as f:
			comments_str = "%" + comments.replace("\n", "\n%")
			f.write(comments_str + "\n" + pgf_str)

		st.text("PGF file written to {}".format(pgf_path_metric))