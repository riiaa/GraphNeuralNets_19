from IPython.display import display, HTML
import torch
import sklearn.metrics


def header_str(a_str, n=80):
    """Returns a string formatted as a header."""
    return '{{:=^{:d}}}'.format(n).format(' ' + a_str + ' ')


def header_html(astr, level=1):
    """Returns a string formatted as a HTML header."""
    html_code = '<h{}>{}</h{}>'.format(level, astr, level)
    return display(HTML(html_code))


def clear_torch_model(model):
    del model
    torch.cuda.empty_cache()


def reg_stats(y_true, y_pred):
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    return r2, mae
