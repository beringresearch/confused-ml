import altair as alt
import pandas as pd

def squares(proba, actual, predicted):
    df = pd.DataFrame(proba, copy = True)
    metadata = pd.DataFrame({'predicted':predicted, 'actual':actual})
    for col in df.columns:
        df[col][metadata['actual']!=col] = None

    df['predicted'] = predicted
    df['actual'] = actual
    df.columns = df.columns.astype(str)
    
    df = df.melt(id_vars=['predicted', 'actual'])

    bins = [b/10 for b in range(11)]
    df['bins'] = pd.cut(df['value'], bins=bins, include_lowest=True)
    b = pd.DataFrame({'count' : df.groupby( [ 'bins', 'variable', 'actual', 'predicted'] )['bins'].size()}).reset_index()
    b['bins'] = b['bins'].astype(str)


    chart = alt.Chart(b, title='').mark_bar().encode(
            x=alt.X('count', axis=alt.Axis(
                                   ticks=False,
                                   labels=False,
                                   title='')),
            y=alt.Y('bins:N', title='Prediction Score', sort=alt.EncodingSortField(field='bins', op='sum', order='descending')),
            column=alt.Column('actual', title=''),
            color='predicted:N',
            tooltip=['count', 'predicted']).properties(width=100).configure_axis(grid=False).configure_view(strokeOpacity=0)

    return chart