import gradio as gr
import plotly.express as px
import pandas as pd

def test_plot_interaction():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "name": ["A", "B", "C"]})
    fig = px.scatter(df, x="x", y="y", hover_name="name")
    
    def on_change(data):
        print(f"DEBUG: Data type: {type(data)}")
        print(f"DEBUG: Data: {data}")
        return str(data)

    with gr.Blocks() as demo:
        plot = gr.Plot(value=fig, label="Interact with me")
        out = gr.Textbox(label="Interaction Data")
        
        plot.change(fn=on_change, inputs=[plot], outputs=[out])
    
    demo.launch()

if __name__ == "__main__":
    test_plot_interaction()
