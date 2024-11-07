import os, sys
import gradio as gr
from src.gradio_demo import SadTalker  


try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False


def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)

def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):

    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<center><span style='font-size: 54px;'>数字人合成——JYD</span></center>")
        gr.Markdown("<center><span style='font-size: 54px;'> </span></center>")
        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('上传形象'):
                        with gr.Column():
                            source_image = gr.Image(label="Source image",type="filepath", elem_id="img2img_image")
                            gr.Examples(
                            ["/opt/jyd01/wangruihua/data/image/a.png",
                             "/opt/jyd01/wangruihua/data/image/b.png"
                             ],
                            [source_image],label='示例形象')

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('上传音频'):
                        with gr.Column():
                            driven_audio = gr.Audio(label="Input audio",type="filepath",elem_id="sadtalker_driven_audio")
                            gr.Examples(
                            ["/opt/jyd01/wangruihua/api/digital/sadtalker/example.wav",
                             ],
                            [driven_audio],label='示例语音')
                            
            

                        # if sys.platform != 'win32' and not in_webui: 
                        #     from src.utils.text2speech import TTSTalker
                        #     tts_talker = TTSTalker()
                        #     with gr.Column(variant='panel'):
                        #         input_text = gr.Textbox(label="Generating audio from text", lines=5, placeholder="please enter some text here, we genreate the audio from text using @Coqui.ai TTS.")
                        #         tts = gr.Button('Generate audio',elem_id="sadtalker_audio_generate", variant='primary')
                        #         tts.click(fn=tts_talker.test, inputs=[input_text], outputs=[driven_audio])
                            
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('参数设置'):
                        gr.Markdown("克隆数字人之前，请先去声音克隆平台克隆声音(http://188.18.18.106:7996) for more detials")
                        with gr.Column(variant='panel'):
                            # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                            # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                            pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0) # 
                            size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model?") # 
                            preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='crop', label='preprocess', info="How to handle input image?")
                            is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")
                            batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2)
                            enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')
                            
                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4")

        if warpfn:
            submit.click(
                        fn=warpfn(sad_talker.test), 
                        inputs=[source_image,
                                driven_audio,
                                preprocess_type,
                                is_still_mode,
                                enhancer,
                                batch_size,                            
                                size_of_image,
                                pose_style
                                ], 
                        outputs=[gen_video]
                        )
        else:
            submit.click(
                        fn=sad_talker.test, 
                        inputs=[source_image,
                                driven_audio,
                                preprocess_type,
                                is_still_mode,
                                enhancer,
                                batch_size,                            
                                size_of_image,
                                pose_style
                                ], 
                        outputs=[gen_video]
                        )

    return sadtalker_interface
 

if __name__ == "__main__":

    demo = sadtalker_demo()
    demo.queue()
    demo.launch(server_port=7996, server_name='0.0.0.0',max_threads=3)


