from gradio_client import Client

client = Client("https://myshell-ai-openvoice.hf.space/--replicas/uh6tf/")
result = client.predict(
		"Howdy!",	# str  in 'Text Prompt' Textbox component
		"default,default",	# str (Option from: [('default', 'default'), ('whispering', 'whispering'), ('cheerful', 'cheerful'), ('terrified', 'terrified'), ('angry', 'angry'), ('sad', 'sad'), ('friendly', 'friendly')]) in 'Style' Dropdown component
		"https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",	# str (filepath on your computer (or URL) of file) in 'Reference Audio' Audio component
		True,	# bool  in 'Agree' Checkbox component
		fn_index=1
)
print(result)
