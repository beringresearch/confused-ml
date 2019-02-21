import altair as alt
import pandas as pd
import numpy as np

def tibble(actual, predicted, ncol=10):
    df = []
    for class_name in np.unique(actual):
        actual_mask = actual == class_name
        predicted_f = np.sort(predicted[actual_mask])
        nrow = (len(predicted_f) // ncol)
        padding = predicted_f.shape[0] % ncol
        column_index = list(np.tile(range(ncol), nrow)) 
        column_index = column_index + [range(ncol)[l] for l in range(len(predicted_f) - len(column_index))]
        row_index = list(np.repeat(range(nrow), ncol)) + list(np.repeat(nrow, len(predicted_f) % ncol))
        res = pd.DataFrame({'Predicted Label': predicted_f,
                            'row': column_index,
                            'column': row_index,
                            'Class Index': class_name})
        df += [res]
    df = pd.concat(df)

    chart = alt.Chart(df).mark_circle().encode(
    x=alt.X('row:O', axis=alt.Axis(
                                   ticks=False,
                                   labels=False,
                                   title='')),
    y=alt.Y('column:O', axis=alt.Axis(
                                   ticks=False,
                                   labels=False,
                                   title='')),
    column=alt.Column('Class Index', title=''),
    color='Predicted Label:N',
    tooltip=['Predicted Label']).properties(width=ncol*6, height=nrow*8).configure_axis(grid=False, domainWidth=0).configure_view(strokeOpacity=0)

    return chart
