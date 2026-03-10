import gradio as gr
import cv2
from PIL import Image, ImageDraw
import os
import subprocess
import shutil
import time
import json

root_path = os.path.dirname(os.path.abspath(__file__))

# Global variable to store the original frame
original_frame = {
    "view1": None,
    "view2": None,
    "view3": None,
    "view4": None
}
prompt = {
    "view1": None,
    "view2": None,
    "view3": None,
    "view4": None
}

def get_first_frame(video_file, view):
    # Extract the first frame
    global original_frame

    if video_file is None:
        original_frame[view] = None
        return None

    cap = cv2.VideoCapture(video_file)
    success, frame = cap.read()
    cap.release()

    if not success:
        original_frame[view] = None
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_frame[view] = Image.fromarray(frame_rgb)
    return original_frame[view]


def click_event(img, view, evt: gr.SelectData):
    # Draw a point prompt
    global original_frame

    if original_frame[view] is None:
        return None

    image = original_frame[view].copy()
    draw = ImageDraw.Draw(image)

    x, y = evt.index
    r = 10
    draw.ellipse((x - r, y - r, x + r, y + r), fill="red", outline="white", width=2)
    prompt[view] = (x, y)
    print(f"{view} clicked at: ({x}, {y})")

    return image


def mka_pipeline(video1, video2, video3, video4, json_file, sam_file):
    timestamp = time.time()
    print(str(timestamp))
    cache_dir = os.path.dirname(video1)
    local_workspace = os.path.join(root_path, "sample_video", str(timestamp))
    local_results = os.path.join(root_path, "results", str(timestamp))
    os.makedirs(local_workspace)
    os.makedirs(local_results)
    
    for file_item in [video1, video2, video3, video4, json_file] :
        shutil.copy(
            file_item, 
            os.path.join(local_workspace, os.path.basename(file_item))
        )
    
    if sam_file is not None:
        shutil.copy(
            sam_file, 
            os.path.join(local_workspace, os.path.basename(sam_file))
        )
    else:
        video_files = [video1, video2, video3, video4]
        view_names = ["view1", "view2", "view3", "view4"]
        sam_prompts = {}
        for video, view in zip(video_files, view_names) :
            sam_prompts[os.path.basename(video)] = (
                {"point": prompt[view]} if prompt[view] is not None else {}
            )
        with open(os.path.join(local_workspace, "sam_prompt.json"), 'w') as sf:
            json.dump(sam_prompts, sf, indent=4)
    
    try:
        subprocess.run(
            ["bash", "run_pipeline.sh", str(timestamp)],
            check=True
        )
        result_video = os.path.join(
            local_results, "pack", "render_smplx_with_mano_sam.mp4"
        )
        shutil.copy(result_video, os.path.join(cache_dir, "render_result.mp4"))
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"
    finally:
        shutil.rmtree(local_workspace)
        shutil.rmtree(local_results)
    
    return os.path.join(cache_dir, "render_result.mp4")


# Gradio UI
with gr.Blocks(
    title="MKA - Markerless Kinematic Analysis", 
    css="""
    .title {
        text-align: center;
        font-size: 2.5em !important;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    .description {
        font-size: 1.1em !important;
        line-height: 1.6;
        text-align: justify;
        padding: 20px;
        background: #f9f9f9;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }
    """
    ) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            "<div class='title'>MKA: Markerless Kinematic Analysis</div>"
        )
        gr.Markdown(
            "We introduce Markerless Kinematic Analysis (MKA), an end-to-end framework" 
            " that reconstructs full-body, articulated 3D meshes, including hands and"
            " manipulated objects, from ordinary RGB videos captured with single or"
            " multiple consumer cameras. Multi-view fusion and explicit human-object"
            " interaction modeling yield anatomically consistent, metric-scale poses"
            " that generalize to cluttered homes, gyms, and clinics. By merging"
            " computer vision with rehabilitative medicine, MKA enables continuous,"
            " objective, and scalable motion monitoring in natural environments,"
            " opening avenues for personalized training, tele-rehabilitation,"
            " and population-level musculoskeletal health surveillance."
        )

    with gr.Row():
        with gr.Column():
            video1 = gr.Video(label="Upload a video from view 1")
        with gr.Column():
            image1 = gr.Image(
                label="Click to prompt",
                interactive=True
            )
    with gr.Row():
        with gr.Column():
            video2 = gr.Video(label="Upload a video from view 2")
        with gr.Column():
            image2 = gr.Image(
                label="Click to prompt",
                interactive=True
            )
    with gr.Row():
        with gr.Column():
            video3= gr.Video(label="Upload a video from view 3")
        with gr.Column():
            image3 = gr.Image(
                label="Click to prompt",
                interactive=True
            )
    with gr.Row():
        with gr.Column():
            video4= gr.Video(label="Upload a video from view 4")
        with gr.Column():
            image4 = gr.Image(
                label="Click to prompt",
                interactive=True
            )
    with gr.Row():
        with gr.Column():
            json_file = gr.File(label="Camera Config JSON")
        with gr.Column():
            sam_file = gr.File(label="SAM Prompt JSON (Optional)")
    
    view1_text = gr.Textbox("view1", visible=False)
    view2_text = gr.Textbox("view2", visible=False)
    view3_text = gr.Textbox("view3", visible=False)
    view4_text = gr.Textbox("view4", visible=False)

    video1.change(fn=get_first_frame, inputs=[video1, view1_text], outputs=image1)
    video2.change(fn=get_first_frame, inputs=[video2, view2_text], outputs=image2)
    video3.change(fn=get_first_frame, inputs=[video3, view3_text], outputs=image3)
    video4.change(fn=get_first_frame, inputs=[video4, view4_text], outputs=image4)

    image1.select(fn=click_event, inputs=[image1, view1_text], outputs=image1)
    image2.select(fn=click_event, inputs=[image2, view2_text], outputs=image2)
    image3.select(fn=click_event, inputs=[image3, view3_text], outputs=image3)
    image4.select(fn=click_event, inputs=[image4, view4_text], outputs=image4)
    
    run_btn = gr.Button("Submit")
    output_video = gr.Video(label="Result")
    
    run_btn.click(
        fn=mka_pipeline,
        inputs=[video1, video2, video3, video4, json_file, sam_file],
        outputs=output_video
    )

if __name__ == "__main__":
    demo.launch(share=True)
