from original import *
import shutil, glob
from easyfuncs import download_from_url, CachedModels, whisperspeak, whisperspeak_on
os.makedirs("dataset",exist_ok=True)
model_library = CachedModels()

with gr.Blocks(title="🔊 RVC Inference",theme=gr.themes.Base()) as app:
    with gr.Row():
        gr.Markdown("# RVC")
        with gr.TabItem("Inference"):
            with gr.Row():
                voice_model = gr.Dropdown(label="Model Voice", choices=sorted(names), value=lambda:sorted(names)[0] if len(sorted(names)) > 0 else '', interactive=True)
                refresh_button = gr.Button("Refresh", variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="Speaker ID",
                    value=0,
                    visible=False,
                    interactive=True,
                )
                vc_transform0 = gr.Number(
                    label="Pitch", 
                    value=0
                )
                but0 = gr.Button(value="Convert", variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("Upload"):
                            dropbox = gr.File(label="Drop your audio here & hit the Reload button.")
                        with gr.TabItem("Record"):
                            record_button=gr.Audio(source="microphone", label="OR Record audio.", type="filepath")
                        
                    with gr.Row():
                        paths_for_files = lambda path:[os.path.abspath(os.path.join(path, f)) for f in os.listdir(path) if os.path.splitext(f)[1].lower() in ('.mp3', '.wav', '.flac', '.ogg')]
                        input_audio0 = gr.Dropdown(
                            label="Input Path",
                            value=paths_for_files('audios')[0] if len(paths_for_files('audios')) > 0 else '',
                            choices=paths_for_files('audios'), # Only show absolute paths for audio files ending in .mp3, .wav, .flac or .ogg
                            allow_custom_value=True
                        )
                    with gr.Row():
                        audio_player = gr.Audio(label="Input")
                        input_audio0.change(
                            inputs=[input_audio0],
                            outputs=[audio_player],
                            fn=lambda path: {"value":path,"__type__":"update"} if os.path.exists(path) else None
                        )
                        record_button.stop_recording(
                            fn=lambda audio:audio, #TODO save wav lambda
                            inputs=[record_button], 
                            outputs=[input_audio0])
                        dropbox.upload(
                            fn=lambda audio:audio.name,
                            inputs=[dropbox], 
                            outputs=[input_audio0])
                        
                with gr.Column():
                    with gr.Accordion("Change Index", open=False):
                        file_index2 = gr.Dropdown(
                            label="Change Index",
                            choices=sorted(index_paths),
                            interactive=True,
                            value=sorted(index_paths)[0] if len(sorted(index_paths)) > 0 else ''
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Index Strength",
                            value=0.5,
                            interactive=True,
                        )
                    vc_output2 = gr.Audio(label="Output")
                    with gr.Accordion("General Settings", open=False):
                        f0method0 = gr.Radio(
                            label="Method",
                            choices=["pm", "harvest", "crepe", "rmvpe"]
                            if config.dml == False
                            else ["pm", "harvest", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label="Breathiness Reduction (Harvest only)",
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Resample",
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Volume Normalization",
                            value=0,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Breathiness Protection (0 is enabled, 0.5 is disabled)",
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        if voice_model != None: vc.get_vc(voice_model.value,protect0,protect0)
                    file_index1 = gr.Textbox(
                        label="Index Path",
                        interactive=True,
                        visible=False#Not used here
                    )
                    refresh_button.click(
                        fn=change_choices,
                        inputs=[],
                        outputs=[voice_model, file_index2],
                        api_name="infer_refresh",
                    )
                    refresh_button.click(
                        fn=lambda:{"choices":paths_for_files('audios'),"__type__":"update"}, #TODO check if properly returns a sorted list of audio files in the 'audios' folder that have the extensions '.wav', '.mp3', '.ogg', or '.flac'
                        inputs=[],
                        outputs = [input_audio0],   
                    )
                    refresh_button.click(
                        fn=lambda:{"value":paths_for_files('audios')[0],"__type__":"update"} if len(paths_for_files('audios')) > 0 else {"value":"","__type__":"update"}, #TODO check if properly returns a sorted list of audio files in the 'audios' folder that have the extensions '.wav', '.mp3', '.ogg', or '.flac'
                        inputs=[],
                        outputs = [input_audio0],   
                    )
            with gr.Row():
                f0_file = gr.File(label="F0 Path", visible=False)
            with gr.Row():
                vc_output1 = gr.Textbox(label="Information", placeholder="Welcome!",visible=False)
                but0.click(
                    vc.vc_single,  
                    [
                        spk_item,
                        input_audio0,
                        vc_transform0,
                        f0_file,
                        f0method0,
                        file_index1,
                        file_index2,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                    ],
                    [vc_output1, vc_output2],
                    api_name="infer_convert",
                )  
                voice_model.change(
                    fn=vc.get_vc,
                    inputs=[voice_model, protect0, protect0],
                    outputs=[spk_item, protect0, protect0, file_index2, file_index2],
                    api_name="infer_change_voice",
                )
        with gr.TabItem("Download Models"):
            with gr.Row():
                url_input = gr.Textbox(label="URL to model", value="",placeholder="https://...", scale=6)
                name_output = gr.Textbox(label="Save as", value="",placeholder="MyModel",scale=2)
                url_download = gr.Button(value="Download Model",scale=2)
                url_download.click(
                    inputs=[url_input,name_output],
                    outputs=[url_input],
                    fn=download_from_url,
                )
            with gr.Row():
                model_browser = gr.Dropdown(choices=list(model_library.models.keys()),label="OR Search Models (Quality UNKNOWN)",scale=5)
                download_from_browser = gr.Button(value="Get",scale=2)
                download_from_browser.click(
                    inputs=[model_browser],
                    outputs=[model_browser],
                    fn=lambda model: download_from_url(model_library.models[model],model),
                )




    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
