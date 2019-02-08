import altair as alt
import pandas as pd

def squares(proba, actual, predicted):
    df = pd.DataFrame(proba)
    metadata = pd.DataFrame({'predicted':predicted, 'actual':actual})
    for col in df.columns:
        df[col][metadata['actual']!=col] = None

    df['predicted'] = predicted
    df['actual'] = actual
    df.columns = df.columns.astype(str)
    
    df = df.melt(id_vars=['predicted', 'actual'])
    alt.data_transformers.enable('default', max_rows=None)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('count(value)', axis=alt.Axis(
                                   ticks=False,
                                   labels=False,
                                   title='')),
        y=alt.Y('value', bin=alt.Bin(maxbins=10), title='Prediction Score'),
        column='actual',
        color='predicted:N',
        tooltip=['count(predicted)', 'predicted']).properties(width=200).configure_axis(grid=False).configure_view(strokeOpacity=0)

    return chart