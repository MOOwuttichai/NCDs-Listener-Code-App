from dash import Dash, html, Input, Output, callback,dcc,dash_table,no_update,dash,State
import pandas as pd
import plotly.express as px
import os
import dash_bootstrap_components as dbc
import dash_daq as daq
import ast
css_tab_STYLE = {'width': '45%','display': 'inline-block','font-family':'THSarabunNew','font-size': '16px','margin-left': '3%','margin-top': '2%'}
css_tab_STYLE_v2 = {'width': '90%','font-family':'THSarabunNew','font-size': '16px','margin-left': '3%'}
css_pie_STYLE = {'width': '90%','margin-left': '3%','font-family':'THSarabunNew','font-size': '20px','align-items': 'center', 'justify-content': 'center'}
css_pie_STYLE_not_show = {'width': '90%','margin-left': '3%','font-family':'THSarabunNew','font-size': '20px','align-items': 'center', 'justify-content': 'center',"display": "none"}
css_bar_and_line_STYLE_not_show={'align-items': 'center', 'justify-content': 'center','font-family':'THSarabunNew','font-size': '20px',"display": "none"}
css_bar_and_line_STYLE={'align-items': 'center', 'justify-content': 'center','font-family':'THSarabunNew','font-size': '20px'}
css_slider_style = {'width': '90%', 'display': 'inline-block','font-family':'THSarabunNew','font-size': '20px','margin-left': '3%'}
css_grahp_name_style={'text-align': 'center','font-family':'THSarabunNew','font-size': '20px'}
css_summare_text = {'text-align': 'left','font-size': '20px','font-family':'THSarabunNew','width': 'auto','margin-left': '9%','margin-top': '9%'}
css_summare_text_v2 = {'text-align': 'left','whiteSpace': 'pre-line','font-size': '20px','font-family':'THSarabunNew','width': 'auto','margin-left': '9%','margin-top': '9%'}
css_summare_text_v = {'box-shadow':' 0 4px 8px 0 rgba(0, 0, 0, 0.2)','border-radius': '18px','backgroundColor':'#d0f8ce','margin-left': '5%'}
css_label_fillter = {'font-size': '18px','text-shadow': '0 0 6px #66B032','font-weight': 'bold'}
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout =  html.Div([
html.Div([
        html.Div([
            html.H1(["     "]),
        html.P('เรียงลำดับข้อมูลในตารางตาม "จำนวนคำ" : ',style={'display': 'inline-block','font-family':'THSarabunNew','font-size': '25px'}),
        dcc.Dropdown(id='soft_table',options=['มากไปน้อย','น้อยไปมาก'],value ='มากไปน้อย',style={'width': '300px','display': 'inline-block','margin-left': 10,'font-size': '20px','font-family':'THSarabunNew'}),
        ],style={'margin-left': '25%'}),
        html.Div([
        dcc.Interval( id="interval",n_intervals=0,),
        dcc.Interval( id="interval_v_o",n_intervals=0,),
        html.Div(id='datatable',style={'height': '350px','overflowY': 'auto','margin-top': 12}),
        ]),
html.H1("Dashboard",style={'text-align': 'center','font-family':'THSarabunNew'}),
html.Div([
    html.H4("พื้นที่เลือกแผนภูมิ", style={'text-align': 'center', 'margin-bottom': '10px'}),  # เพิ่มหัวข้อ

    dbc.ButtonGroup([
        dbc.Button("เพศผู้ป่วย", id="Gender-button", n_clicks=0, color="success", className="me-1"),
        dbc.Button("โรคที่กล่าวถึง", id="diseases_type-button", n_clicks=0, color="success", className="me-1"),
        dbc.Button("อาการที่กล่าวถึง", id="symptom_type-button", n_clicks=0, color="success", className="me-1"),
        dbc.Button("วิธีการรักษา", id="Treatment_type-button", n_clicks=0, color="success", className="me-1"),
        dbc.Button("พฤติกรรมผู้ป่วย", id="behaviour_type-button", n_clicks=0, color="success", className="me-1"),
    ], className="mb-2"),

    dbc.ButtonGroup([
        dbc.Button("ลักษณะความคิดเห็น", id="comment_type-button", n_clicks=0, color="danger", className="me-1"),
        dbc.Button("ประสบการณ์", id="exp-button", n_clicks=0, color="danger", className="me-1"),
        dbc.Button("คำที่พบบ่อย", id="top_five_word-button", n_clicks=0, color="danger", className="me-1"),
    ], className="mb-2"),
], style={'text-align': 'center'}),
dbc.Container([
            html.Div([
                html.Div([
                        html.P("ความมีประโยชน์:",style=css_label_fillter),
                        dcc.Checklist(id="pie-charts-useful-names"),
                        ],style=css_tab_STYLE),
                html.Div([
                    html.P("ประสบการณ์:",style=css_label_fillter),
                    dcc.Checklist(id='pie-charts-exp-names')
                    ],style=css_tab_STYLE),
                html.Div([
                    html.P("เพศ:",style=css_label_fillter),
                    dcc.Checklist(id='pie-charts-Gender-names')
                    ],style=css_tab_STYLE),
                html.Div([
                html.P("โรค:",style=css_label_fillter),
                    dcc.Dropdown(id='pie-charts-cancer-names',multi=True)
                    ],style=css_tab_STYLE_v2),
                html.Div([
                    html.P("อาการ:",style=css_label_fillter),
                    dcc.Dropdown(id='pie-charts-sym-names',multi=True)
                    ],style=css_tab_STYLE_v2),
                html.Div([
                    html.P("วิธีการรักษา:",style=css_label_fillter),
                    dcc.Dropdown(id='pie-charts-TH-names',multi=True)
                    ],style=css_tab_STYLE_v2),
                html.Div([
                    html.P("พฤติกรรมผู้ป่วย:",style=css_label_fillter),
                    dcc.Dropdown(id='pie-charts-BEH-names',multi=True)
                    ],style=css_tab_STYLE_v2),
                html.Div([
                    html.P("จำนวนคำที่ต้องการหา:",style=css_label_fillter),
                    dcc.Dropdown(['1 คำ','2 คำ','3 คำ'],'2 คำ',id='pie-word-n-g-names')
                    ],style=css_tab_STYLE_v2),
                # html.Div([
                #     html.P("จำนวนคำในประโยค(ขั้นต่ำ):",style=css_label_fillter),
                #     dcc.Slider(0,200,200/5,value=0,
                #         tooltip={"placement": "bottom", "always_visible": True},
                #         id='slider-count_word-names'),
                #     ],style=css_slider_style),
                html.Div([
                    html.P("จำนวนการตอบกลับ(ขั้นต่ำ):",style=css_label_fillter),
                    dcc.Slider(0, 100, 100/5,
                        value=0,tooltip={"placement": "bottom", "always_visible": True},
                        id='slider-count_rechat-names'),
                    ],style=css_slider_style),
                html.Div([
                    html.P("จำนวน like(ขั้นต่ำ):",style=css_label_fillter),
                    dcc.Slider(0, 500, 500/5,
                        value=0,tooltip={"placement": "bottom", "always_visible": True},
                        id='slider-count_like-names'),
                    ],style=css_slider_style),
                html.Div([
                        html.P("จำนวนคำ 'ขั้นต่ำ':",style=css_label_fillter),
                        dcc.Slider(0, 500, 500/5,
                        value=0,tooltip={"placement": "bottom", "always_visible": True},
                        id='my-numeric-input-1'),
                        ],style=css_slider_style),
                        # dcc.Input(id='my-numeric-input-1',type= "number",placeholder="จำนวนคำ 'ขั้นต่ำ' ที่มีประโยชน์",value = 3)
            ],style={
                'width': 300,
                'margin-left': 20,
                'margin-top': 35,
                'margin-bottom': 35,
                'height':1000,
                'backgroundColor':'#d0f8ce',
                'overflowY': 'auto',
                'border-radius': '18px',
                'margin-right': '12px',
                'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
                }),
    # html.Div([
    #     dbc.Tabs([settings_tab, charts_tab]),
    #     ]),
    html.Div([
    dbc.ModalBody([
        html.Div([
            html.H3("แผนภูมิ เพศผู้ป่วยกับจำนวนความคิดเห็น",id="title_Gender",style=css_grahp_name_style),
            dcc.Graph(id="pie-charts-Gender-graph",style=css_pie_STYLE),
            html.Div([
            html.P(id = 'text_sum_Gender',style=css_summare_text)
            ],style=css_summare_text_v),
        ],style={'width': 'auto','grid-area': 'gender'}),
        html.Div([
            html.H3("แผนภูมิ โรคกับจำนวนความคิดเห็น",id="title_carcer",style=css_grahp_name_style),
            dcc.Graph(id="pie-charts-carcer-graph",style=css_tab_STYLE_v2),
            html.Div([
            html.P(id = 'text_sum_cancer',style=css_summare_text_v2)
            ],style=css_summare_text_v),
        ],style={'width': 'auto','grid-area': 'carcer'}),
        html.Div([
            html.H3("แผนภูมิ อาการกับจำนวนความคิดเห็น",id="title_sym",style=css_grahp_name_style),
            dcc.Graph(id="his-charts-sym-graph",style=css_tab_STYLE_v2),
            html.Div([
            html.P(id ="text_sum_sym",style=css_summare_text_v2)
            ],style=css_summare_text_v),
        ],style={'width': 'auto','grid-area': 'sym'}),
        # html.Div([
        #     html.H3("แผนภูมิ จำนวนคำกับชื่อผู้ที่มาเเสดงความคิดเห็น",style=css_grahp_name_style),
        #     dcc.Graph(id="line-charts-count_word-graph",style=css_bar_and_line_STYLE),
        #     # html.Div([
        #     # html.P(id ="text_sum_word",style=css_summare_text_v2)
        #     # ],style=css_summare_text_v),
        # ],style={'width': 'auto'}),
        html.Div([
            html.H3("แผนภูมิ วิธีการรักษากับจำนวนความคิดเห็น",id="title_thet",style=css_grahp_name_style),
            dcc.Graph(id="line-charts-like-graph",style=css_bar_and_line_STYLE),
            html.Div([
            html.P(id ="text_sum_T",style=css_summare_text_v2)
            ],style=css_summare_text_v),
        ],style={'width': 'auto','grid-area': 'thet'}),
        html.Div([
            html.H3("แผนภูมิ จำนวนความคิดเห็นกับพฤติกรรมผู้ป่วย",id="title_behav",style=css_grahp_name_style),
            dcc.Graph(id="line-B-P-graph",style=css_bar_and_line_STYLE),
            html.Div([
            html.P(id ="text_sum_like",style=css_summare_text_v2)
            ],style=css_summare_text_v),
        ],style={'width': 'auto','grid-area': 'behav'}),
        html.Div([]),
        #--------------------------กราฟที่ไม่มีการแสดงให้เห็น-------------------------------#
        html.Div([
            html.H3("แผนภูมิ คำที่ปรากฏมากที่สุด 5 อันดับแรก",id="title_word",style=css_grahp_name_style),
            dcc.Graph(id="line-charts-rechat-graph",style=css_bar_and_line_STYLE_not_show),
            html.Div([
            html.P(id ="text_sum_fiveo",style=css_summare_text_v2)
            ],style=css_summare_text_v),
        ],style={'width': 'auto','grid-area': 'word_top'}),
         html.Div([
            html.H3("แผนภูมิ ลักษณะของความคิดเห็นกับจำนวนความคิดเห็น",id="title_tyar",style=css_grahp_name_style),
            dcc.Graph(id="pie-charts-useful-graph",style=css_pie_STYLE_not_show),
            html.Div([
            html.P(id ="text_sum_useful",style=css_summare_text),
            ],style=css_summare_text_v),
        ],style={'width': 'auto','grid-area': 'type_c'}),
        html.Div([
            html.H3("แผนภูมิ การเล่าประสบการณ์กับจำนวนความคิดเห็น",id="title_exp",style=css_grahp_name_style),
            dcc.Graph(id="pie-charts-exp-graph",style=css_pie_STYLE_not_show),
            html.Div([
            html.P(id = 'text_sum_exp',style=css_summare_text)
            ],style=css_summare_text_v),
        ],style={'width': 'auto','grid-area': 'exp_c'}),
        #--------------------------พวกไม่มีกราฟ-------------------------------#
        html.Div([
            html.H3("จำนวนคำที่ มากสุด น้อยสุด และเฉลี่ย",style=css_grahp_name_style),
            html.Div([
            html.P(id ="text_sum_BEH",style=css_summare_text_v2)
            ],style=css_summare_text_v),
        ],style={'width': 'auto','grid-area': 'sum_beh'}),
        html.Div([
            html.H3("จำนวนการกดถูกใจที่ มากสุด น้อยสุด และเฉลี่ย",style=css_grahp_name_style),
            html.Div([
            html.P(id ="text_sum_reply",style=css_summare_text_v2)
            ],style=css_summare_text_v),
        ],style={'width': 'auto','grid-area': 'sum_reply'}),
        html.Div([
            html.H3("จำนวนตอบกลับที่ มากสุด น้อยสุด และเฉลี่ย",style=css_grahp_name_style),
            html.Div([
            html.P(id ="text_sum_word",style=css_summare_text_v2)
            ],style=css_summare_text_v),
        ],style={'width': 'auto','grid-area': 'sum_word'}),
        ],id="grid-print-area",style={'display':'grid', 'grid-template-areas': '"gender" "carcer" "sym" "thet" "behav" "word_top" "type_c" "exp_c" "sum_beh" "sum_reply" "sum_word"', 'grid-gap': '20px'})
    ],
        style={
            'width': 1300,
            'height':1000,
            'margin-top': 35,
            'margin-right': 10,
            'margin-bottom': 35,
            'box-shadow':' 0 4px 8px 0 rgba(0, 0, 0, 0.2)',
            'border-radius': '18px',
            'overflowY': 'auto',
            'display':'flex'
        }),html.Div(id="dummy"),
]   ,fluid=True,
    style={'display': 'flex'},),
html.Div([
dbc.Button("Print", id="grid-browser-print-btn"),
# ,style={'height': '60px','width': '200px','textAlign': 'center'
#                                                         ,'background-color': '#04AA6D','color':' white','border-radius': '12px',
#                                                         'align-items': 'center','font-size': '24px','display': 'flex',
#                                                         'justify-content': 'center','margin-left': '43%','margin-top': 30,'border-color':'white',
#                                                         'box-shadow': '2px 2px 20px 10px #7fecad'}
]),                                                            
]),

html.Div([dbc.Container([
        dbc.Card([
            dbc.CardBody([
                html.H1('Summary of Insights from Data Extracted by Google Gemini', 
                        style={'text-align': 'center', 'font-family': 'THSarabunNew'}),
                html.P(id="text_sum_bybot", 
                       style={'whiteSpace': 'pre-line', 'font-family': 'THSarabunNew', 'font-size': '25px', 'height': 'auto'}),  # ปรับความสูงเป็น auto
            ]),
        ], style={'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)', 'border-radius': '18px', 'background-color': '#d0f8ce'}),  # ปรับสีพื้นหลัง
    ], style={'margin-top': '10px', 'margin-bottom': '10px'}),])
])
# คำสั่งซ่อน/แสดงของ เพศ
@callback(
    Output("pie-charts-Gender-graph", "style"),
    Output("Gender-button", "style"),
    Output("title_Gender", "style"),
    Output("text_sum_Gender", "style"),
    Input("Gender-button", "n_clicks"),
    State("pie-charts-Gender-graph", "style"),
    State("title_Gender", "style"),
    State("text_sum_Gender", "style"),
)
def exp_plot(n_clicks, current_style, title_style, text_style):
    if n_clicks is None or n_clicks % 2 == 0:
        return {"display": "block"}, {'background-color': 'green'}, css_grahp_name_style, css_summare_text # show all
    else:
        return {"display": "none"}, {'background-color': 'red'}, {'display': 'none'}, {'display': 'none'} # hide graph, title, text, keep button shown
    
# คำสั่งซ่อน/แสดงของ โรคที่ถูกกล่าวถึง
@callback(
    Output("pie-charts-carcer-graph", "style"),
    Output("diseases_type-button", "style"),
    Output("title_carcer", "style"),
    Output("text_sum_cancer", "style"),
    Input("diseases_type-button", "n_clicks"),
    State("pie-charts-carcer-graph", "style"),
    State("title_carcer", "style"),
    State("text_sum_cancer", "style"),     
)
def exp_plot(n_clicks, current_style, title_style, text_style):
    if n_clicks % 2 == 0:
        return {"display": "block"},{'background-color': 'green'},css_grahp_name_style,css_summare_text_v2  # Show plot
    else:
        return {"display": "none"},{'background-color': 'red'}, {'display': 'none'}, {'display': 'none'}  # Hide plot

# คำสั่งซ่อน/แสดงของ อาการที่ถูกกล่าวถึง
@callback(
    Output("his-charts-sym-graph", "style"),
    Output("symptom_type-button", "style"),
    Output("title_sym", "style"),
    Output("text_sum_sym", "style"),
    Input("symptom_type-button", "n_clicks"),
    State("his-charts-sym-graph", "style"),
    State("title_sym", "style"),
    State("text_sum_sym", "style"),     
)
def exp_plot(n_clicks, current_style, title_style, text_style):
    if n_clicks % 2 == 0:
        return {"display": "block"},{'background-color': 'green'} ,css_grahp_name_style,css_summare_text_v2  # Show plot
    else:
        return {"display": "none"},{'background-color': 'red'}, {'display': 'none'}, {'display': 'none'}  # Hide plot
    
# คำสั่งซ่อน/แสดงของ วิธีการรักษาที่ถูกกล่าวถึง
@callback(
    Output("line-charts-like-graph", "style"),
    Output("Treatment_type-button", "style"),
    Output("title_thet", "style"),
    Output("text_sum_T", "style"),
    Input("Treatment_type-button", "n_clicks"),
    State("line-charts-like-graph", "style"),
    State("title_thet", "style"),
    State("text_sum_T", "style"),     
)
def exp_plot(n_clicks, current_style, title_style, text_style):
    if n_clicks % 2 == 0:
        return {"display": "block"},{'background-color': 'green'},css_grahp_name_style,css_summare_text_v2  # Show plot
    else:
        return {"display": "none"} ,{'background-color': 'red'} , {'display': 'none'}, {'display': 'none'}# Hide plot
    
# คำสั่งซ่อน/แสดงของ พฤติกรรมของผู้ป่วย
@callback(
    Output("line-B-P-graph", "style"),
    Output("behaviour_type-button", "style"),
    Output("title_behav", "style"),
    Output("text_sum_like", "style"),
    Input("behaviour_type-button", "n_clicks"),
    State("line-B-P-graph", "style"),
    State("title_behav", "style"),
    State("text_sum_like", "style"),     
)
def exp_plot(n_clicks, current_style, title_style, text_style):
    if n_clicks % 2 == 0:
        return {"display": "block"},{'background-color': 'green'},css_grahp_name_style,css_summare_text_v2  # Show plot
    else:
        return {"display": "none"},{'background-color': 'red'}, {'display': 'none'}, {'display': 'none'}  # Hide plot

# คำสั่งซ่อน/แสดงของ ลักษณะความคิดเห็น
@callback(
    Output("pie-charts-useful-graph", "style"),
    Output("comment_type-button", "style"),
    Output("title_tyar", "style"),
    Output("text_sum_useful", "style"),
    Input("comment_type-button", "n_clicks"),
    State("pie-charts-useful-graph", "style"),
    State("title_tyar", "style"),
    State("text_sum_useful", "style"),     
)
def comment_type_plot(n_clicks, current_style, title_style, text_style):
    if n_clicks % 2 == 0:
        return {"display": "none"},{'background-color': 'red'}, {'display': 'none'}, {'display': 'none'} # Show plot
    else:
        return {"display": "block"},{'background-color': 'green'},css_grahp_name_style, css_summare_text # Hide plot
# คำสั่งซ่อน/แสดงของ คำที่พบบ่อยที่สุด
@callback(
    Output("line-charts-rechat-graph", "style"),
    Output("top_five_word-button", "style"),
    Output("title_word", "style"),
    Output("text_sum_fiveo", "style"),
    Input("top_five_word-button", "n_clicks"),
    State("line-charts-rechat-graph", "style"),
    State("title_word", "style"),
    State("text_sum_fiveo", "style"),     
)
def top_five_word_plot(n_clicks, current_style, title_style, text_style):
    if n_clicks % 2 == 0:
        return {"display": "none"},{'background-color': 'red'},{'display': 'none'}, {'display': 'none'}  # Show plot
    else:
        return {"display": "block"},{'background-color': 'green'},css_grahp_name_style,css_summare_text_v2  # Hide plot
# คำสั่งซ่อน/แสดงของ ประสบการณ์
@callback(
    Output("pie-charts-exp-graph", "style"),
    Output("exp-button", "style"),
    Output("title_exp", "style"),
    Output("text_sum_exp", "style"),
    Input("exp-button", "n_clicks"),
    State("pie-charts-exp-graph", "style"),
    State("title_exp", "style"),
    State("text_sum_exp", "style"),     
)
def exp_plot(n_clicks, current_style, title_style, text_style):
    if n_clicks % 2 == 0:
        return {"display": "none"},{'background-color': 'red'},{'display': 'none'}, {'display': 'none'} # Show plot
    else:
        return {"display": "block"},{'background-color': 'green'},css_grahp_name_style,css_summare_text   # Hide plot

@callback(
    Output(component_id='interval', component_property='interval'),
    Input("interval", "n_intervals"),
)
def inten_n (n):
    time_run = pd.read_csv('time_run.csv')
    if time_run.iloc[0,0] == 0:
        time_run.iloc[0,0] = 1
        time_run.to_csv('time_run.csv',index=False)
        return 5*1000
    else:
        return 20*1000
# พื้นที่ปรับแต่ง Dashboard กำหนดว่าจะให้มีค่าอะไรบ้าง
@callback(
    Output('pie-charts-exp-names', "options",allow_duplicate=True),
    Output('pie-charts-exp-names', "value",allow_duplicate=True),
    Output('pie-charts-cancer-names', "options",allow_duplicate=True),
    Output('pie-charts-cancer-names', "value",allow_duplicate=True),
    Output('pie-charts-Gender-names', "options",allow_duplicate=True),
    Output('pie-charts-Gender-names', "value",allow_duplicate=True),
    Output('pie-charts-useful-names', "options",allow_duplicate=True),
    Output('pie-charts-useful-names', "value",allow_duplicate=True),
    Output('pie-charts-sym-names', "options",allow_duplicate=True),
    Output('pie-charts-TH-names', "options",allow_duplicate=True),
    Output('pie-charts-BEH-names', "options",allow_duplicate=True),
    Input("interval", "n_intervals"),
    prevent_initial_call=True,
)
def input_tag(n):
    import dash_bootstrap_components as dbc
    data_for_dash_facebook = pd.read_csv('data_for_dash_01.csv', encoding='utf-8-sig')
    sym_o_th = data_for_dash_facebook.iloc[:, 16:]
    sym_o1_th = sym_o_th.melt()
    sym_o2_th = (pd.crosstab(sym_o1_th['variable'], sym_o1_th['value']).rename(columns={0: 'ไม่มีการเล่า', 1: 'มีการเล่า'})).reset_index()
    o_1=data_for_dash_facebook['defind_exp_with_python'].unique()
    v_1=data_for_dash_facebook['defind_exp_with_python'].unique()
    o_2=data_for_dash_facebook['defind_cancer_with_nlp'].unique()
    v_2=data_for_dash_facebook['defind_cancer_with_nlp'].unique()
    o_3 = data_for_dash_facebook['defind_Genden_with_python'].unique()
    v_3 = data_for_dash_facebook['defind_Genden_with_python'].unique()
    o_4 = data_for_dash_facebook['label'].unique()
    v_4 = data_for_dash_facebook['label'].unique()
    o_5 = sym_o2_th['variable'].unique()
    #วิธีการรักษา
    comment_data_T = data_for_dash_facebook[['name','Treatment']]
    fix_word = []
    for i in comment_data_T['Treatment']:
        x = ast.literal_eval(i)
        fix_word.append(x)
    comment_data_T['Treatment']=fix_word
    Treatment_dumm = comment_data_T['Treatment'].str.join(sep='*').str.get_dummies(sep='*')
    comment_data_T_1 = comment_data_T.join(Treatment_dumm)
    sym_c = comment_data_T_1.iloc[:, 2:]
    sym_ca = sym_c.melt()
    sym_can = (pd.crosstab(sym_ca['variable'], sym_ca['value']).rename(columns={0: 'ไม่มีการเล่า', 1: 'มีการเล่า'})).reset_index()
    sym_can = sym_can[sym_can['มีการเล่า']!=0]
    o_6 = sym_can['variable'].unique()
    #พฤติกรรม
    comment_data_B = data_for_dash_facebook[['name','beha']]
    fix_word = []
    for i in comment_data_B['beha']:
        x = ast.literal_eval(i)
        fix_word.append(x)
    comment_data_B['beha']=fix_word
    B_dumm = comment_data_B['beha'].str.join(sep='*').str.get_dummies(sep='*')
    comment_data_B_1 = comment_data_B.join(B_dumm)
    sym_B = comment_data_B_1.iloc[:, 2:]
    sym_Ba = sym_B.melt()
    sym_Bcan = (pd.crosstab(sym_Ba['variable'], sym_Ba['value']).rename(columns={0: 'ไม่มีการเล่า', 1: 'มีการเล่า'})).reset_index()
    sym_Bcan = sym_Bcan[sym_Bcan['มีการเล่า']!=0]
    o_7 = sym_Bcan['variable'].unique()
    return (o_1,v_1,o_2,v_2,o_3,v_3,o_4,v_4,o_5,o_6,o_7) #,v_6,

@callback(
    Output("pie-charts-exp-graph", "figure",allow_duplicate=True),
    Output('pie-charts-Gender-graph', "figure",allow_duplicate=True),
    Output("pie-charts-carcer-graph", "figure",allow_duplicate=True),
    Output("pie-charts-useful-graph", "figure",allow_duplicate=True),
    Output("his-charts-sym-graph", "figure",allow_duplicate=True),
    Output("line-charts-like-graph", "figure",allow_duplicate=True),
    Output("line-charts-rechat-graph","figure",allow_duplicate=True),
    Output("line-B-P-graph","figure",allow_duplicate=True),
    # component_id='bar-graph-matplotlib', component_property='src'
    # Output("line-charts-count_word-graph", "figure",allow_duplicate=True), 
    Output("text_sum_exp", 'children'),
    Output("text_sum_Gender", 'children'),
    Output("text_sum_cancer", 'children'),
    Output("text_sum_useful", 'children'), 
    Output("text_sum_sym", 'children'),
    Output("text_sum_T", 'children'),
    Output("text_sum_fiveo", 'children'),
    Output("text_sum_like", 'children'),
    Output("text_sum_reply", 'children'),
    Output("text_sum_word", 'children'),
    Output("text_sum_BEH", 'children'),
    Output('datatable', 'children',allow_duplicate=True),
    Output("text_sum_bybot", 'children'),
    Input("interval", "n_intervals"),
    Input("pie-charts-exp-names", "value"),
    Input('pie-charts-Gender-names', "value"),
    Input('pie-charts-cancer-names', "value"),
    Input('pie-charts-useful-names', "value"),
    Input("pie-charts-sym-names", "value"),
    # Input("slider-count_word-names", "value"),
    Input("slider-count_like-names", "value"),
    Input("slider-count_rechat-names", "value"),
    Input('my-numeric-input-1', 'value'),
    Input('soft_table', 'value'),
    Input('pie-word-n-g-names', 'value'),
    Input("pie-charts-TH-names", "value"),
    Input("pie-charts-BEH-names", "value"),
    prevent_initial_call=True)

def generate_chart(n,exp,Gender,carcer,useful,sym,count_like,count_rechat,real_useFul,soft_table,n_gam,THEM,BEH): #,count_word
    import dash_bootstrap_components as dbc
    data_for_dash_facebook = pd.read_csv('data_for_dash_01.csv', encoding='utf-8-sig')
    data_for_dash_facebook['count_plot'] = 1
    nms = data_for_dash_facebook
    if sym == [] or sym is None:
            nms = data_for_dash_facebook
    else:
        for defind_sym in range(len(sym)):
            x = nms[nms[sym[defind_sym]]==1]
            nms = x
    # real_useFul_1=[]
    # for token in nms['จำนวนคำ']:
    #     if token >= real_useFul:
    #         real_useFul_1.append('อาจมีประโยชน์')
    #     else:
    #         real_useFul_1.append('ไม่มีประโยชน์')
    # nms['use_ful'] = real_useFul_1
    nms = nms[nms['defind_exp_with_python'].isin(exp)]
    nms = nms[nms['defind_Genden_with_python'].isin(Gender)]
    nms = nms[nms['defind_cancer_with_nlp'].isin(carcer)]
    nms = nms[nms['label'].isin(useful)]
    print(nms.iloc[:, 16:])
    sym_c = nms.iloc[:, 16:-1]
    sym_ca = sym_c.melt()
    sym_can = (pd.crosstab(sym_ca['variable'], sym_ca['value']).rename(columns={0: 'ไม่มีการเล่า', 1: 'มีการเล่า'}))
    # print(sym_can)
    plot_data=sym_can['มีการเล่า'].reset_index()
    if sym == [] or sym is None :
        plot_data_None = plot_data
        plot_sym=plot_data_None[plot_data_None['มีการเล่า'] != 0]
    elif len(sym) == 1:
        plot_data_None = plot_data
        plot_sym=plot_data_None[plot_data_None['มีการเล่า'] != 0]
    else:
        x_value=plot_data[plot_data['variable'].isin(sym)]['มีการเล่า'].to_list()
        x_name=plot_data[plot_data['variable'].isin(sym)]['variable'].to_list()
        sum_name =""
        for sum_name_count in x_name:
            if sum_name == "":
                sum_name = sum_name+sum_name_count
            else:
                sum_name = sum_name+"&"+sum_name_count
        x_plot_1 = pd.DataFrame(data={'variable': sum_name, 'มีการเล่า': x_value})
        plot_data_non=plot_data[plot_data.variable.isin(sym) == False]
        plot_data_nonnone=pd.concat([plot_data_non,x_plot_1])
        plot_sym=plot_data_nonnone.drop_duplicates()
        plot_sym = plot_sym[plot_sym['มีการเล่า']!=0]
    nms = nms[nms['จำนวนคำ']>=real_useFul]
    nms = nms[nms['ยอดไลค์']>=count_like]
    nms = nms[nms['จำนวนการตอบกลับ']>=count_rechat]
    #=================================================================================
    comment_data_T = nms[['name','Treatment']]
    fix_word = []
    import ast
    for i in comment_data_T['Treatment']:
        x = ast.literal_eval(i)
        fix_word.append(x)
    comment_data_T['Treatment']=fix_word
    Treatment_dumm = comment_data_T['Treatment'].str.join(sep='*').str.get_dummies(sep='*')
    comment_data_T_1 = comment_data_T.join(Treatment_dumm)
    if THEM == [] or THEM is None:
            nms = nms
    else:
        for defind_THEM in range(len(THEM)):
            x = comment_data_T_1[comment_data_T_1[THEM[defind_THEM]]==1]
        target_names = x['name'].tolist()
        nms = nms[nms['name'].isin(target_names)]
#================================================================================================
    fix_word = []
    comment_data_B = nms[['name','beha']]
    import ast
    for i in comment_data_B['beha']:
        x = ast.literal_eval(i)
        fix_word.append(x)
    comment_data_B['beha']=fix_word
    B_dumm = comment_data_B['beha'].str.join(sep='*').str.get_dummies(sep='*')
    comment_data_B_1 = comment_data_B.join(B_dumm)
    if BEH == [] or BEH is None:
            nms = nms
    else:
        for defind_BEH in range(len(BEH)):
            x = comment_data_B_1[comment_data_B_1[BEH[defind_BEH]]==1]
        target_names = x['name'].tolist()
        nms = nms[nms['name'].isin(target_names)]  
#==============================================================================================
    #---------------------------------------
    from nltk.probability import FreqDist
    import ast
    fix_word = []
    for i in range(len(nms)):
        try:
            x = ast.literal_eval(nms['token'][i])
            # print(x)
            fix_word.append(x)
            nms['token']=fix_word
        except:
            pass
    
    def remove_words(word_list, words_to_remove):
        for word in words_to_remove:
            while word in word_list:
                word_list.remove(word)
        return word_list

    # Example usage (assuming 'sum_word' is your list)
    if n_gam == '1 คำ':
        n_gam_use = 'token'
    elif n_gam == '2 คำ':
        n_gam_use = 'token_2g'
    elif n_gam == '3 คำ':
        n_gam_use = 'token_3g'
    from itertools import chain
    sum_word_10 = []
    fix_word = []
    try:
        for i in (nms[f'{n_gam_use}']):
            x = ast.literal_eval(i)
            fix_word.append(x)
        nms[f'{n_gam_use}']=fix_word
    except:
        pass
    for i in nms[f'{n_gam_use}']:
        x = i
        sum_word_10+=x
        sum_word_10 = sum_word_10[:-1]
    fdist_1= FreqDist(sum_word_10)
    r=list(nms['defind_cancer_with_nlp'].unique())
    for i in r:
        if i in fdist_1:  # Check if the key exists
            del fdist_1[i]
    x=fdist_1.most_common(5)
    word_count_table_ST = pd.DataFrame(x,columns=['word','count'])
#----------------------------------------------------------------------------------
#----------------------------------------
# วิธีการรักษา
    sym_c = comment_data_T_1.iloc[:, 2:]
    sym_ca = sym_c.melt()
    sym_can = (pd.crosstab(sym_ca['variable'], sym_ca['value']).rename(columns={0: 'ไม่มีการเล่า', 1: 'มีการเล่า'}))
    plot_data_T=sym_can['มีการเล่า'].reset_index()
    if THEM == [] or THEM is None :
        plot_sym_1 = plot_data_T[plot_data_T['มีการเล่า']!=0]
        plot_sym_1 = plot_sym_1[plot_sym_1['variable']!='ไม่สามารถระบุได้']
    elif len(THEM) == 1:
        plot_sym_1 = plot_data_T[plot_data_T['มีการเล่า']!=0]
        plot_sym_1 = plot_sym_1[plot_sym_1['variable']==THEM[0]]
    else:
        x_value=plot_data_T[plot_data_T['variable'].isin(THEM)]['มีการเล่า'].to_list()
        x_name=plot_data_T[plot_data_T['variable'].isin(THEM)]['variable'].to_list()
        sum_name =""
        for sum_name_count in x_name:
            if sum_name == "":
                sum_name = sum_name+sum_name_count
            else:
                sum_name = sum_name+"&"+sum_name_count
        x_plot_1 = pd.DataFrame(data={'variable': sum_name, 'มีการเล่า': x_value})
        plot_data_non=plot_data[plot_data.variable.isin(THEM) == False]
        plot_data_nonnone=pd.concat([plot_data_non,x_plot_1])
        plot_sym_1=plot_data_nonnone.drop_duplicates()
        plot_sym_1 = plot_sym_1[plot_sym_1['มีการเล่า']!=0]
        plot_sym_1 = plot_sym_1[plot_sym_1['variable']!='ไม่สามารถระบุได้']
# พฤติกรรม
    sym_B = comment_data_B_1.iloc[:, 2:]
    sym_Ba = sym_B.melt()
    sym_Bcan = (pd.crosstab(sym_Ba['variable'], sym_Ba['value']).rename(columns={0: 'ไม่มีการเล่า', 1: 'มีการเล่า'}))
    plot_data_B=sym_Bcan['มีการเล่า'].reset_index()
    if BEH == [] or BEH is None :
        plot_sym_1_B = plot_data_B[plot_data_B['มีการเล่า']!=0]
        plot_sym_1_B = plot_sym_1_B[plot_sym_1_B['variable']!='ไม่สามารถระบุได้']
    elif len(BEH) == 1:
        plot_sym_1_B = plot_data_B[plot_data_B['มีการเล่า']!=0]
        plot_sym_1_B = plot_sym_1_B[plot_sym_1_B['variable']==BEH[0]]
    else:
        x_value=plot_data_B[plot_data_B['variable'].isin(BEH)]['มีการเล่า'].to_list()
        x_name=plot_data_B[plot_data_B['variable'].isin(BEH)]['variable'].to_list()
        sum_name =""
        for sum_name_count in x_name:
            if sum_name == "":
                sum_name = sum_name+sum_name_count
            else:
                sum_name = sum_name+"&"+sum_name_count
        x_plot_1 = pd.DataFrame(data={'variable': sum_name, 'มีการเล่า': x_value})
        plot_data_non=plot_data[plot_data.variable.isin(BEH) == False]
        plot_data_nonnone=pd.concat([plot_data_non,x_plot_1])
        plot_sym_1_B=plot_data_nonnone.drop_duplicates()
        plot_sym_1_B = plot_sym_1_B[plot_sym_1_B['มีการเล่า']!=0]
        plot_sym_1_B = plot_sym_1_B[plot_sym_1_B['variable']!='ไม่สามารถระบุได้']

#คำนวณ % ให้กราฟ แท่ง
    total_comments = len(nms)
    # โรค
    counts_dise = nms['defind_cancer_with_nlp'].value_counts().reset_index()
    counts_dise['precent'] = ((counts_dise['count']/ len(nms['defind_cancer_with_nlp'])) * 100).round(2)
    percentages_dise = counts_dise
    # อาการ
    plot_sym['Percentage'] = ((plot_sym['มีการเล่า'] / total_comments) * 100).round(2)
    # วิธีการรักษา
    plot_sym_1['Percentage'] = ((plot_sym_1['มีการเล่า'] / total_comments) * 100).round(2)
    print(percentages_dise)
    # พฤติกรรม
    plot_sym_1_B['Percentage'] = ((plot_sym_1_B['มีการเล่า'] / total_comments) * 100).round(2)
# สร้างกราฟ
    fig_1 = px.pie(nms, values='count_plot', names=nms['defind_exp_with_python'],labels={'defind_exp_with_python':'ถูกเล่าจาก ','count_plot':'จำนวนความคิดเห็น '},hole=.5,color='defind_exp_with_python',color_discrete_map={'ไม่สามารถระบุได้':'lightcyan','เล่าประสบการณ์ตัวเอง':"royalblue",'เล่าประสบการณ์คนอื่น':'darkblue'}) #,color_discrete_sequence=px.colors.sequential.Teal
    fig_2 = px.pie(nms, values='count_plot', names=nms['defind_Genden_with_python'],hole=.5,color='defind_Genden_with_python',labels={'defind_Genden_with_python':'Gender ','count_plot':'จำนวนความคิดเห็น '},color_discrete_sequence=px.colors.sequential.Emrld)#,color='defind_Genden_with_python',color_discrete_map={'เพศชาย':'#4424D6','เพศหญิง':"#C21460",'ไม่ระบุเพศ':'#DAD4F7'}
    fig_3 = px.bar(percentages_dise,y='precent', x='defind_cancer_with_nlp',hover_data={'defind_cancer_with_nlp': True, 'count': True, 'precent': False},text_auto='auto',labels={'defind_cancer_with_nlp':'โรค ','count':'จำนวนความคิดเห็น '},color='defind_cancer_with_nlp',color_discrete_sequence=px.colors.qualitative.Light24)#,color='defind_cancer_with_nlp'
    fig_4 = px.pie(nms, values='count_plot', names=nms['label'],color='label',color_discrete_sequence=px.colors.sequential.Hot,)
    fig_5 = px.bar(plot_sym, x='variable', y='Percentage', color='variable', hover_data={'variable': True, 'มีการเล่า': True, 'Percentage': False}, text_auto=True,color_discrete_sequence=px.colors.qualitative.Alphabet_r,labels={'variable': 'อาการ' ,'มีการเล่า': 'จำนวนความคิดเห็น'})
    fig_6 = px.bar(plot_sym_1, y='variable', x='Percentage', color='variable', hover_data={'variable': True, 'มีการเล่า': True, 'Percentage': False},color_discrete_sequence=px.colors.sequential.Viridis, text_auto=True, labels={'variable': 'วิธีการรักษา', 'มีการเล่า': 'จำนวนความคิดเห็น'})
    fig_7 = px.bar(plot_sym_1_B, y='variable', x='Percentage', color='variable', hover_data={'variable': True, 'มีการเล่า': True, 'Percentage': False},color_discrete_sequence=px.colors.sequential.Viridis, text_auto=True, labels={'variable': 'พฤติกรรม', 'มีการเล่า': 'จำนวนความคิดเห็น'})
    fig_G1 = px.bar(word_count_table_ST, y='count', x='word',text_auto=True,color_discrete_sequence=['#8601AF'],width= 600,)
# ปรับแต่งกราฟ 
    fig_G1.update_layout(xaxis_title="Number of Comments",yaxis_title="Words found",plot_bgcolor="#FFFFFF",font=dict(size=17),yaxis_range=[0,max(word_count_table_ST['count'])+50])
    fig_G1.update_traces(textposition='outside')
    fig_1.update_layout(legend=dict( orientation="h"),annotations=[dict(text=f'total: {len(nms)}', x=sum(fig_1.get_subplot(1, 1).x) / 2, y=0.5,
                      font_size=20, showarrow=False, xanchor="center")])
    fig_1.update_traces(textposition='inside', textinfo='percent',textfont_size=30)
    fig_2.update_layout(legend=dict( orientation="h"),annotations=[dict(text=f'total: {len(nms)}', x=sum(fig_1.get_subplot(1, 1).x) / 2, y=0.5,
                      font_size=20, showarrow=False, xanchor="center")])
    fig_2.update_traces(textposition='inside', textinfo='percent',textfont_size=15) # ,marker=dict(line=dict(color='#000000', width=2))
    
    if len(percentages_dise) > 1:
        max_max_dise = [0, max(percentages_dise['precent'])+10]
    else:
        max_max_dise = [0,100+10]
    
    if len(plot_sym) > 1:
        max_max_plot_sym = [0, max(plot_sym['Percentage'])+10]
    else:
        max_max_plot_sym = [0,100+10]

    if len(plot_sym_1) > 1:
        max_max_plot_sym_1 = [0, max(plot_sym_1['Percentage'])+10]
    else:
        max_max_plot_sym_1 = [0,100+10]

    if len(plot_sym_1) > 1:
        print(plot_sym_1)
        max_max_plot_sym_1_B = [0, max(plot_sym_1_B['Percentage'])+10]
    else:
        max_max_plot_sym_1_B = [0,100+10]
        
    fig_3.update_layout(
    font=dict(
        size=15,
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
    )
    fig_3.update_traces(marker_line_width=1.5, opacity=1)#
    fig_3.update_layout(xaxis_title="Number of Comments",yaxis_title="Diseases found",plot_bgcolor="#FFFFFF",yaxis_range=max_max_dise)
    fig_3.update_traces(textposition='outside')
    fig_3.update_traces(texttemplate="%{y} %",textposition='outside') 
    fig_4.update_layout(legend=dict( orientation="h"))
    # fig_5.update_layout(xaxis_title="อาการที่พบ",yaxis_title="จำนวนความคิดเห็น",plot_bgcolor="#D9EEBF")
    fig_5.update_layout(xaxis_title="อาการที่พบ", yaxis_title="จำนวนความคิดเห็น", plot_bgcolor="#FFFFFF",yaxis_range=max_max_plot_sym)
    fig_5.update_traces(marker_line_width=1.5, opacity=1)
    fig_5.update_layout(xaxis_title="Number of Comments",yaxis_title="symptoms found",plot_bgcolor="#FFFFFF")
    fig_5.update_traces(textposition='outside')
    fig_5.update_traces(texttemplate="%{y} %",textposition='outside') 
    fig_5.update_layout(
        font=dict(
            size=17,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig_6.update_layout(xaxis_title="Treatment methods", yaxis_title="Number of Comments", plot_bgcolor="#FFFFFF",xaxis_range=max_max_plot_sym_1)
    fig_6.update_traces(marker_line_width=1.5, opacity=1)
    fig_6.update_traces(textposition='outside')
    fig_6.update_traces(texttemplate="%{x} %",textposition='outside') 
    fig_7.update_layout(xaxis_title="พฤติกรรมของผู้ป่วย", yaxis_title="Number of Comments", plot_bgcolor="#FFFFFF",xaxis_range=max_max_plot_sym_1_B)
    fig_7.update_traces(marker_line_width=1.5, opacity=1)
    fig_7.update_traces(textposition='outside')#
    fig_7.update_traces(texttemplate="%{x} %",textposition='outside') 
    fig_1.update_layout(clickmode='event+select')
    fig_2.update_layout(clickmode='event+select')
    fig_3.update_layout(clickmode='event+select')
    fig_4.update_layout(clickmode='event+select')
    fig_5.update_layout(clickmode='event+select')

    fig_3.update_layout(showlegend=False)
    fig_5.update_layout(showlegend=False)
    # สรุปกราฟประสบการณ์
    sum_all_exp = nms['defind_exp_with_python']
    sum_non_exp = nms[nms['defind_exp_with_python'].isin(['ไม่ได้เล่าประสบการณ์',"Didn't tell the experience"]) ]
    sum_other_exp = nms[nms['defind_exp_with_python'].isin(['เล่าประสบการณ์คนอื่น',"Tell other people's experiences"])]
    sum_self_exp = nms[nms['defind_exp_with_python'].isin(['เล่าประสบการณ์ตัวเอง','Tell about your own experiences'])] 
    value_non_p = (len(sum_non_exp)/len(sum_all_exp))*100
    value_other_p = (len(sum_other_exp)/len(sum_all_exp))*100
    value_self_p = (len(sum_self_exp)/len(sum_all_exp))*100
    text_1 = f'''ผลจากแผนภูมิ ประสบการณ์กับความคิดเห็น พบว่า ความคิดเห็นที่ไม่ได้เล่าประสบการณ์เกี่ยวกับโรคมีจำนวน {len(sum_non_exp)} ความคิดเห็น คิดเป็นร้อยละ {round(value_non_p, 2)} โดยความคิดเห็นที่เป็นการเล่าประสบการณ์เกี่ยวกับโรคสำหรับหัวข้อนี้ส่วนมากเป็น 
    { f"การเล่าประสบการณ์คนอื่นจำนวน {len(sum_other_exp)} ความคิดเห็นคิดเป็นร้อยละ {round(value_other_p,2)} ในขณะที่อีก {len(sum_self_exp)} ความคิดเห็นคิดเป็นร้อยละ {round(value_self_p,2)} เป็นการเล่าจากประสบการณ์ตนเอง" 
    if len(sum_other_exp) > len(sum_self_exp) else 
    f"การเล่าประสบการณ์ตัวเองจำนวน {len(sum_self_exp)} ความคิดเห็นคิดเป็นร้อยละ {round(value_self_p,2)} ในขณะที่อีก {len(sum_other_exp)} ความคิดเห็นคิดเป็นร้อยละ {round(value_other_p,2)} เป็นการเล่าจากประสบการณ์คนอื่น"}'''
    #สรุปกราฟเพศ 
    sum_all_gen = nms['defind_Genden_with_python']
    sum_female_gen = nms[nms['defind_Genden_with_python'].isin(['เพศหญิง','Female'])]
    sum_male_gen = nms[nms['defind_Genden_with_python'].isin(['เพศชาย','Male'])]
    sum_non_gen = nms[nms['defind_Genden_with_python'].isin(['ไม่ระบุเพศ','Gender not specified'])]
    value_female_p_gen = (len(sum_female_gen)/len(sum_all_gen))*100
    value_male_p_gen = (len(sum_male_gen)/len(sum_all_gen))*100
    value_non_p_gen = (len(sum_non_gen)/len(sum_all_gen))*100
    text_2 = f'''ผลจากแผนภูมิ เพศกับความคิดเห็น พบว่า ความคิดเห็นที่ไม่สามารถระบุเพศได้มีจำนวน {len(sum_non_gen)} ความคิดเห็น คิดเป็นร้อยละ {round(value_non_p_gen, 2)} โดยความคิดเห็นที่มีการระบุเพศ สำหรับหัวข้อนี้ส่วนมากเป็น 
    { f"เพศชายจำนวน {len(sum_male_gen)} ความคิดเห็น คิดเป็นร้อยละ {round(value_male_p_gen,2)} ในขณะที่อีก {len(sum_female_gen)} ความคิดเห็น คิดเป็นร้อยละ {round(value_female_p_gen,2)} เป็นเพศหญิง" 
        if len(sum_male_gen) > len(sum_female_gen) else 
        f"เพศหญิงจำนวน {len(sum_female_gen)} ความคิดเห็น คิดเป็นร้อยละ {round(value_female_p_gen,2)} ในขณะที่อีก {len(sum_male_gen)} ความคิดเห็น คิดเป็นร้อยละ {round(value_male_p_gen,2)} เป็นเพศชาย"}'''
    # สรุปกราฟโรค
    nms_can = nms[['defind_cancer_with_nlp','count_plot']]
    nms_gro_cancer = nms_can.groupby('defind_cancer_with_nlp').sum().reset_index().sort_values(by='count_plot',ascending=False)
    text_3 ='โรคทั้งหมดที่พบในหัวข้อนี้ได้เเก่ \n'
    for fill in range(len(nms_gro_cancer)):
        p_can = (nms_gro_cancer["count_plot"][fill]/nms_gro_cancer["count_plot"].sum())*100
        text_3 = text_3 + f'- {nms_gro_cancer["defind_cancer_with_nlp"][fill]} มีจำนวน {nms_gro_cancer["count_plot"][fill]} ความคิดเห็น คิดเป็นร้อยละ {round(p_can,2)}\n'
    # สรุปกราฟมีประโยชน์หรือไม่มีประโยชน์
    sum_all_gen_label = nms['label']
    sum_t_exp_gen = nms[nms['label'].isin(['tell experience',"เล่าประสบการณ์"])]
    sum_que_gen = nms[nms['label'].isin(['Question',"คำถาม"])]
    sum_useless_gen = nms[nms['label'].isin(['useless/unimportant',"ไม่มีประโยชน์/ไม่สำคัญ"])]
    value_t_exp_p_gen = (len(sum_t_exp_gen)/len(sum_all_gen_label))*100
    value_que_p_gen = (len(sum_que_gen)/len(sum_all_gen_label))*100
    value_useless_p_gen = (len(sum_useless_gen)/len(sum_all_gen_label))*100
    text_4 = f'''ผลจากแผนภูมิ ลักษณธความคิดเห็น พบว่า ความคิดเห็นที่ไม่สามารถระบุประโยชน์ได้มีจำนวน {len(sum_useless_gen)} ความคิดเห็น คิดเป็นร้อยละ {round(value_useless_p_gen, 2)} โดยความคิดเห็นที่มีการระบุลักษณะ สำหรับหัวข้อนี้ส่วนมากเป็น 
    { f"ความคิดเห็นที่เล่าประสบการณ์จำนวน {len(sum_t_exp_gen)} ความคิดเห็น คิดเป็นร้อยละ {round(value_t_exp_p_gen,2)} ในขณะที่อีก {len(sum_que_gen)} ความคิดเห็น คิดเป็นร้อยละ {round(value_que_p_gen,2)} เป็นความคิดเห็นลักษณะคำถาม" 
        if len(sum_t_exp_gen) > len(sum_que_gen) else 
        f"ความคิดเห็นลักษณะคำถามจำนวน {len(sum_que_gen)} ความคิดเห็น คิดเป็นร้อยละ {round(value_que_p_gen,2)} ในขณะที่อีก {len(sum_t_exp_gen)} ความคิดเห็น คิดเป็นร้อยละ {round(value_t_exp_p_gen,2)} เป็นความคิดเห็นที่เล่าประสบการณ์"}'''
    #อาการ
    text_5 ='อาการทั้งหมดที่พบในหัวข้อนี้ได้เเก่ \n'
    for filler in range(len(plot_sym)):
        word_sym_next = plot_sym.reset_index()
        text_5 = text_5 + f'- {word_sym_next["variable"][filler]} มีจำนวน {word_sym_next["มีการเล่า"][filler]} ความคิดเห็น \n'
    # like reply จำนวนคำ
    text_6 ='วิธีการรักษาทั้งหมดที่พบในหัวข้อนี้ได้เเก่ \n'
    for filler in range(len(plot_sym_1)):
        plot_sym_next_T = plot_sym_1.reset_index()
        text_6 = text_6 + f'- {plot_sym_next_T["variable"][filler]} มีจำนวน {plot_sym_next_T["มีการเล่า"][filler]} ความคิดเห็น \n'
    text_7 ='คำที่พบมากที่สุด 5 อันดับแรกในหัวข้อนี้ได้เเก่ \n'
    for filler in range(len(word_count_table_ST)):
        plot_sym_next_W = word_count_table_ST.reset_index()
        text_7 = text_7 + f'- {plot_sym_next_W["word"][filler]} มีจำนวน {plot_sym_next_W["count"][filler]} ความคิดเห็น \n'
    text_8 ='พฤติกรรมที่พบในหัวข้อนี้ได้เเก่ \n'
    for filler in range(len(plot_sym_1_B)):
        B_sym_next = plot_sym_1_B.reset_index()
        text_8 = text_8 + f'- {B_sym_next["variable"][filler]} มีจำนวน {B_sym_next["มีการเล่า"][filler]} ความคิดเห็น \n'

    avg_11=nms['ยอดไลค์'].mean()
    avg_22=nms['จำนวนการตอบกลับ'].mean()
    avg_33=nms['จำนวนคำ'].mean()
    text_9 = f'''จากแผนภูมิจะพบว่า ยอดไลค์ในหัวข้อนี้มีค่ามากที่สุด คือ {nms['ยอดไลค์'].max()}
                ยอดไลค์ในหัวข้อนี้มีค่าน้อยที่สุด คือ {nms['ยอดไลค์'].min()}
                ยอดไลค์ในหัวข้อนี้มีค่าเฉลี่ยที่สุด คือ {round(avg_11,2)}'''
    text_10= f'''จากแผนภูมิจะพบว่า จำนวนการตอบกลับในหัวข้อนี้มีค่ามากที่สุด คือ {nms['จำนวนการตอบกลับ'].max()}
                จำนวนการตอบกลับในหัวข้อนี้มีค่าน้อยที่สุด คือ {nms['จำนวนการตอบกลับ'].min()}
                จำนวนการตอบกลับในหัวข้อนี้มีค่าเฉลี่ยที่สุด คือ {round(avg_22,2)}'''
    text_11= f'''จากแผนภูมิจะพบว่า จำนวนคำในหัวข้อนี้มีค่ามากที่สุด คือ {nms['จำนวนคำ'].max()}
                จำนวนคำในหัวข้อนี้มีค่าน้อยที่สุด คือ {nms['จำนวนคำ'].min()}
                จำนวนคำในหัวข้อนี้มีค่าเฉลี่ยที่สุด คือ {round(avg_33,2)}'''
    if soft_table == 'มากไปน้อย':
        nms = nms.sort_values(by='จำนวนคำ',ascending=False)
        if nms['ยอดไลค์'].sum() > 1:
            # data130 = nms.iloc[:,:4]
            data130 = nms.iloc[:,1:5]
        else :
            # data130 = nms.iloc[:,:2]
            data130 = nms.iloc[:,1:3]
    else:
        nms = nms.sort_values(by='จำนวนคำ',ascending=True)
        if nms['ยอดไลค์'].sum() > 1:
            # data130 = nms.iloc[:,:4]
            data130 = nms.iloc[:,1:5]
        else :
            # data130 = nms.iloc[:,:2]
            data130 = nms.iloc[:,1:3]
    data_for_export = dash_table.DataTable(data130.to_dict('records'), [{"name": i, "id": i} for i in data130.columns],style_cell={'textAlign': 'left','font_size': '10px'}
                                           ,style_header={'backgroundColor': '#04AA6D','color': 'white'},export_format="csv")
    with open('bot_summarize_comment.txt', 'r',encoding='utf-8-sig') as file:
        look_orgin = file.read()
        look = look_orgin.replace('*', '')
    return [fig_1,fig_2,fig_3,fig_4,fig_5,fig_6,fig_G1,fig_7,text_1,text_2,text_3,text_4,text_5,text_6,text_7,text_8,text_9,text_10,text_11,data_for_export,look]#
    # ,fig_7,fig_8,text_7,text_8,
app.clientside_callback(
            """
            function () {            

                var printContents = document.getElementById('grid-print-area').innerHTML;
                var originalContents = document.body.innerHTML;

                document.body.innerHTML = printContents;

                window.print();

                document.body.innerHTML = originalContents;      
                location.reload()                              

                return window.dash_clientside.no_update
            }
            """,
            Output("dummy", "children"),
            Input("grid-browser-print-btn", "n_clicks"),
            prevent_initial_call=True,
        )

if __name__ == '__main__':
    app.run(debug=True)
    