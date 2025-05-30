import os
import re
import json
import time
import signal
import traceback
from functools import wraps


def extract_generated_code(resp, tokenizer=None):
    if not isinstance(resp, str):
        import vllm  # noqa: 避免全局依赖vllm

    if isinstance(resp, vllm.outputs.RequestOutput):
        # print(resp.prompt[-200:])
        # print('#'*100)
        resp = tokenizer.decode(resp.outputs[0].token_ids)
        resp = resp.strip()
    if not resp.endswith('"""'):
        resp += '"""'
    ans = resp.split('"""')
    return ans[-2].strip().replace('</s>', '')


def read_json_file(file):
    with open(file, 'r', encoding='UTF-8') as file:
        data = json.load(file)
    return data


def read_jsonl_file(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def dump_json_file(obj, file):
    with open(file, 'w', encoding='UTF-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def dump_jsonl_file(records, path):
    with open(path, 'w', encoding='utf-8') as outfile:
        for record in records:
            outline = json.dumps(record, ensure_ascii=False)
            outfile.write(outline + "\n")


def single_plot(x_data, y_data, line_type='散点+折线图', show_img=True,
                x_title='值', y_title='频率', title=None,
                fig_name=None, html_name=None):
    '''简单绘图
    Notes:
        line_type: '散点图', '面积图', '折线图', '散点+折线图'
    '''
    import plotly.express as px         # noqa: 避免全局依赖plotly
    import plotly.graph_objs as go      # noqa: 避免全局依赖plotly

    if title is None:
        title = line_type

    if line_type == '散点图':
        fig = px.scatter(x=x_data, y=y_data, title=title)
    elif line_type == '面积图':
        fig = px.area(x=x_data, y=y_data, title=title)
    elif line_type == '折线图':
        fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='lines', name=title))
    elif line_type == '散点+折线图':
        fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='lines+markers', name=title))
    else:
        fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='lines', name=title))

    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    if show_img:
        fig.show()

    if fig_name:
        fig.write_image(fig_name, format='png', width=800, height=600, scale=3)
    if html_name:
        fig.write_html(html_name)


def multi_plot(x_data, y_data_list, line_type='散点+折线图', show_img=True,
                x_title='值', y_title='频率', name_list=None,
                fig_name=None, html_name=None):
    '''简单绘图
    Notes:
        line_type: '散点图', '面积图', '折线图', '散点+折线图'
    '''
    import plotly.express as px         # noqa: 避免全局依赖plotly
    import plotly.graph_objs as go      # noqa: 避免全局依赖plotly

    line_type_mapping = {'散点+折线图': 'lines+markers', '折线图': 'lines'}
    data = []
    for idx, y_data in enumerate(y_data_list):
        trace = go.Scatter(x=x_data, y=y_data, mode=line_type_mapping[line_type], name=name_list[idx])
        data.append(trace)
    fig = go.Figure(data=data)
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    if show_img:
        fig.show()
    if fig_name:
        fig.write_image(fig_name, format='png', width=800, height=600, scale=3)
    if html_name:
        fig.write_html(html_name)


def insert_data_to_visual_html(data, ori_html='static/visual.html', new_html='static/visual_new.html'):
    '''将数据插入到html中
    Args:
        data: 要插入的数据
        ori_html: 原始html文件
        new_html: 新的html文件
    '''
    ori_html_lines = []
    with open(ori_html, 'r', encoding='UTF-8') as f:
        for line in f:
            ori_html_lines.append(line)

    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)

    # 找到插入位置
    pattern = r'\s+const metric_data = .*;\n'
    for idx in range(len(ori_html_lines)):
        if re.match(pattern, ori_html_lines[idx]):
            prefix_idx = ori_html_lines[idx].find('const metric_data = ')
            new_line = ori_html_lines[idx][:prefix_idx] + "const metric_data = '" + str(data) + "';\n"
            ori_html_lines[idx] = new_line
            break

    with open(new_html, 'w', encoding='UTF-8') as f:
        for line in ori_html_lines:
            f.write(line)


def get_schema_list(path):
    if not os.path.exists(path):
        return []
    else:
        return read_json_file(path)


def get_schema_dict(path):
    if not os.path.exists(path):
        return {}
    else:
        return read_json_file(path)


def gen_idx_sources_dict(path):
    source_dict = {}
    if not os.path.exists(path):
        raise ValueError("Please input the correct path of dataset file.")
    dataset = read_json_file(path)
    for idx, data in enumerate(dataset):
        source_dict[idx] = data['source']
    return source_dict
def gen_idx_sources_dict1(path):
    source_dict = {}
    if not os.path.exists(path):
        raise ValueError("Please input the correct path of dataset file.")
    dataset = read_jsonl_file(path)
    for idx, data in enumerate(dataset):
        source_dict[idx] = data['source']
    return source_dict

def norm_name(name):
    if '-' in name:
        return name.replace('-', ' ').lower()
    elif '_' in name:
        return name.replace('_', ' ').lower()
    else:
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        return name.lower()


class MyTimeOutError(AssertionError):
    def __init__(self, value="Time Out"):
        self.value = value

    def __str__(self):
        return repr(self.value)


def _raise_exception(exception, exception_message):
    if exception_message is None:
        raise exception()
    else:
        raise exception(exception_message)


def timeout(seconds, timeout_exception=MyTimeOutError, exception_message=None):
    """函数超时装饰器, 指定超时时间，超时后抛出异常. NOTE: 这里支持了嵌套的超时设置
    参考自: https://www.saltycrane.com/blog/2010/04/using-python-timeout-decorator-uploading-s3/
    Args:
        seconds: 超时时间，单位秒
        timeout_exception: 超时后抛出的异常
        exception_message: 超时后抛出异常的消息
    Notes:
				1. 已经测试过可以支持多重嵌套（如二重、三重）    
    """
    def decorate(function):

        def handler(signum, frame):
            _raise_exception(timeout_exception, exception_message)

        @wraps(function)
        def new_function(*args, **kwargs):
            if not seconds:
                return function(*args, **kwargs)

            old = signal.signal(signal.SIGALRM, handler)
            old_left_time = signal.getitimer(signal.ITIMER_REAL)

            # 支持嵌套: 设置在之前的时间范围内
            true_seconds = seconds
            if old_left_time[0]:
                true_seconds = min(old_left_time[0], seconds)

            # 使用功能更为强大的setitimer函数, 可以更精确的控制超时时间（相比alarm函数）
            signal.setitimer(signal.ITIMER_REAL, true_seconds)

            start_time = time.time()
            try:
                result = function(*args, **kwargs)
            finally:
                end_time = time.time()
                # 支持嵌套: 恢复之前的剩余超时时间
                old_left_time = max(0, old_left_time[0] - (end_time - start_time))

                signal.setitimer(signal.ITIMER_REAL, old_left_time)
                signal.signal(signal.SIGALRM, old)
            return result

        return new_function

    return decorate