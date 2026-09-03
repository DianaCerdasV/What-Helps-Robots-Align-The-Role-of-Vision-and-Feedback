import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import yaml
import os
from ollama import chat, Client
import re
import ast
import time
from openai import OpenAI
import yamlloader
import sounddevice as sd
import numpy as np
import torch
import whisper
from gtts import gTTS
import playsound
import tempfile
import threading
import base64
from PIL import Image
import io

drives = []
chosen_conversation_file = None
current_dir = os.path.dirname(os.path.abspath(__file__))


def init_conversation_file():
    """Find the first unused conversation file path and store it globally."""
    global chosen_conversation_file
    base_dir = current_dir
    base_name = "conversation_"
    extension = ".yaml"
    i = 1
    while True:
        file_name = f"{base_name}{i}{extension}"
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            chosen_conversation_file = file_path
            break
        i += 1

def save_conversation(conversation):
    """Persist the conversation list to the selected YAML conversation file."""
    if chosen_conversation_file is None:
        raise RuntimeError("init_conversation_file() must be called first")
    with open(chosen_conversation_file, "w", encoding="utf-8") as file:
        yaml.dump(conversation, file, allow_unicode=True)

class LLMmodel:
    """LLM/VLM client wrapper for features in the GUI."""

    def __init__(self, model, initial_prompt):
        """Configure model client instance and hosting provider.

        :param model: Model name or identifier.
        :type model: str
        :param initial_prompt: Initial prompt data used for requests.
        :type initial_prompt: dict or str
        """
        self.model = model
        self.initial_prompt = initial_prompt
        if model == "phi4:14b" or model == "Qwen3:30b":
            self.client = Client(host="http://10.113.36.20:11434")
            self.host = "ollama"
        else:
            # Set your OpenAI / OpenRouter API Key here
            self.client = OpenAI(api_key="API-KEY", base_url="https://openrouter.ai/api/v1")
            self.host = "openrouter"

    @staticmethod
    def encode_image_to_base64(image_path):
        """Encode image file to base64 for vision model prompts."""
        if not image_path or not os.path.exists(image_path):
            return None
        with Image.open(image_path) as img:
            img.thumbnail((800, 800))
            # Ensure image is in RGB mode
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def LLM_worker(client, host, model, command, images=None):
        """Perform a chat request to the configured LLM backend."""

        if host == "ollama":
            resp = client.chat(model=model, messages=command, options={"temperature": 0.1})
            return resp.message.content
        else:
            resp = client.chat.completions.create(model=model, messages=command)
            return resp.choices[0].message.content

    def send_to_LLM(self, conversation, extra_context):
        """Send a text conversation to the model and return the response."""
        messages = [self.initial_prompt]
        messages.extend(extra_context)
        messages.extend(conversation)  
        generated_text = self.LLM_worker(self.client, self.host, self.model, messages)
        return generated_text
    
    def send_to_VLM(self, image_path, context=None):
        """Send an image to the model in vision mode and return text output."""
        if not image_path:
            raise ValueError("Must provide image_path.")
        b64 = self.encode_image_to_base64(image_path)
        if not b64:
            raise ValueError("Could not encode image to base64.")
        system_msg = self.initial_prompt if isinstance(self.initial_prompt, dict) else {"role": "system", "content": str(self.initial_prompt)}

        if self.host == "ollama":
            messages = [system_msg]
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": f"Analyze this image: data:image/jpeg;base64,{b64}"})
            return self.LLM_worker(self.client, self.host, self.model, messages)
        else:
            messages = [system_msg]
            if context:
                messages.extend(context)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    },
                ],
            })
            return self.LLM_worker(self.client, self.host, self.model, messages)


def load_configuration(config_file):
    """Load model configuration from a YAML file.

    :param config_file: Path to a configuration YAML that contains model and prompt.
    :type config_file: str
    :return: Tuple of model name and initial prompt object.
    :rtype: tuple[str, dict]
    """
    if not os.path.isfile(config_file):
        raise FileExistsError(f"{config_file} does not exist")
    config = yaml.load(open(config_file, "r", encoding="utf-8"), Loader=yamlloader.ordereddict.CLoader)
    model = config["model"]
    initial_prompt = config["initial_prompt"]
    return model, initial_prompt

def extract_missions_and_drives(text):
    """Extract missions and drives structures from completion text.

    Expected format in text is:
    Mission1: [tag, value] Drive: ...
    Mission2: [tag, value] Drive: ...

    :param text: Raw response from the LLM, as string or list of strings.
    :type text: str or list
    :return: Tuple containing extracted missions and drives lists.
    :rtype: tuple[list, list]
    """
    if isinstance(text, list):
        text = "\n".join(text)
    missions = []
    drives = []
    mission_blocks = re.findall(r'(Mission\d+:\s*\[.*?\]\s*Drive:\s*.*?)(?=(?:\n\s*\n|$))', text, re.DOTALL)
    for block in mission_blocks:
        tag_match = re.search(r'Mission\d+:\s*\[([^\],]+),\s*([0-9.]+)\]', block)
        drive_match = re.search(r'Drive:\s*(.*)', block)
        if tag_match and drive_match:
            mission_tag = tag_match.group(1)
            mission_value = float(tag_match.group(2))
            drive = drive_match.group(1).strip()
            missions.append([mission_tag, mission_value])
            drives.append(drive)
    return missions, drives

class ChatInterface:
    """Tkinter based chat interface for alignment, mission and drive generation."""

    def __init__(self, root):
        """Initialize GUI, model clients, and conversation context.

        :param root: Tkinter root window.
        :type root: tk.Tk
        """
        self.root = root
        self.root.title("Robot Interaction System")
        self.first_interaction = True

        self.mode = "alignment"
        self.conversation = []
        self.image_path = None
        self.final_purpose = None
        self.final_mission = None
        self.final_drives = None
        self.results = None

        # Load LLMs
        alignment_prompt = os.path.join(current_dir, "humanpurpose_promptOG.yaml")
        mission_prompt = os.path.join(current_dir, "missions_prompt.yaml")
        drives_prompt = os.path.join(current_dir, "drives_promptchanged.yaml")
        needs_file = os.path.join(current_dir, "internal_needs.yaml")
        perceptions_file = os.path.join(current_dir, "objects.yaml")
        VLM_prompt = os.path.join(current_dir, "VLM_prompt_2.yaml")

        model_a, initial_a = load_configuration(alignment_prompt)
        model_m, initial_m = load_configuration(mission_prompt)
        model_d, initial_d = load_configuration(drives_prompt)
        vlm_model, vlm_initial = load_configuration(VLM_prompt)

        self.LLM_alignment = LLMmodel(model_a, initial_a)
        self.LLM_mission = LLMmodel(model_m, initial_m)
        self.LLM_drives = LLMmodel(model_d, initial_d)
        self.VLM = LLMmodel(vlm_model, vlm_initial)

        # Load perceptions and needs once
        needs = yaml.load(open(needs_file, "r", encoding="utf-8"), Loader=yamlloader.ordereddict.CLoader)
        perceptions = yaml.load(open(perceptions_file, "r", encoding="utf-8"), Loader=yamlloader.ordereddict.CLoader)
        self.objects = [{"role": "system", "content": str(perceptions["objects"])}]
        self.needs = [{"role": "system", "content": str(needs["needs"])}]

        # Conversation file
        init_conversation_file()

        # Interface
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=90, height=30, bg="#f4f4f4", font=("Arial", 11))
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.user_input = tk.Text(root, width=70, height=4, font=("Arial", 12))
        self.user_input.grid(row=1, column=0, padx=10, pady=10)
        self.user_input.bind("<Control-Return>", self.send_message)

        self.send_button = tk.Button(root, text="Send", command=self.send_message, width=10, bg="#48C9F1", fg="white")
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.voice_button = tk.Button(root, text="Speak", command=self.record_and_transcribe_voice, width=10, bg="#4CAF50", fg="white")
        self.voice_button.grid(row=2, column=0, pady=(0, 10))

        self.attach_button = tk.Button(root, text="Attach Image", command=self.attach_image, width=12, bg="#888888", fg="white")
        self.attach_button.grid(row=2, column=1, padx=10, pady=(0, 10), sticky="e")

    def display_message(self, role, message):
        """Append a role/message pair to the chat window.

        :param role: Sender role, such as 'user', 'assistant', or 'system'.
        :type role: str
        :param message: Text message to display.
        :type message: str
        """
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{role.capitalize()}: {message}\n\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.yview(tk.END)

    def attach_image(self):
        """Open file dialog to select and attach an image for VLM processing."""
        path = filedialog.askopenfilename(title="Select image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if path and os.path.exists(path):
            self.image_path = path
            self.display_message('system', f"Image attached: {os.path.basename(path)}")

    def send_message(self, event=None):
        """Handle send button event, forward text to alignment model and display output.

        The first interaction may include an attached image processed by VLM.
        """
        user_msg = self.user_input.get("1.0", tk.END).strip()
        if not user_msg:
            return

        self.display_message("user", user_msg)
        self.user_input.delete("1.0", tk.END)

        if self.mode == "alignment":
            if self.first_interaction:
                self.first_interaction = False
                if self.image_path:
                    image_description = self.VLM.send_to_VLM(image_path=self.image_path, context=self.objects)
                    print("Image description:", image_description)
                    self.conversation.append({"role": "user", "content": user_msg + "\n\nImage description: " + image_description})
                else:
                    self.conversation.append({"role": "user", "content": user_msg})
            else: 
                self.conversation.append({"role": "user", "content": user_msg})
            
            reply = self.LLM_alignment.send_to_LLM(self.conversation, self.objects)
            self.image_path = None
            self.conversation.append({"role": "assistant", "content": reply})
            self.display_message("assistant", reply)
            save_conversation(self.conversation)
            self.speak_and_close(reply)

    def generate_missions(self):
        """Request mission generation from the mission-targeted LLM.

        Combines current objects and needs in system context, sends the final purpose
        as user input, and stores the response as final_mission.
        """
        combined_context = [self.objects, self.needs]
        user_input = [{"role": "user", "content": self.final_purpose}]
        reply = self.LLM_mission.send_to_LLM(user_input, combined_context)
        self.conversation.append({"role": "assistant", "content": reply})
        save_conversation(self.conversation)
        print("Generated Missions:", reply)
        self.final_mission = reply
        self.generate_drives()

    def generate_drives(self):
        """Request drive generation from the drives-targeted LLM.

        Uses the final purpose and generated mission text as input.
        """
        input = [{"role": "user", "content": self.final_purpose + "\n\n" + self.final_mission}]
        reply = self.LLM_drives.send_to_LLM(input, self.objects)
        self.conversation.append({"role": "assistant", "content": reply})
        save_conversation(self.conversation)
        print("Generated Drives:", reply)
        self.final_drives = reply
        self.results = reply
        
    def speak_and_close(self, reply):
        """Calls the text-to-speech function and handles final completion signals to trigger mission generation.

        :param reply: Text reply from the alignment model.
        :type reply: str
        """
        self.speak_text(reply)
        if "Final message" in reply or "Final description" in reply:
            self.final_purpose = reply
            self.generate_missions()

    def record_and_transcribe_voice(self):
        """Record spoken audio (mic) and transcribe to text using Whisper."""
        if not hasattr(self, "recording_state"):
            self.recording_state = False
        if not self.recording_state:
            self.recording_state = True
            self.voice_button.config(text="Stop", bg="#E74C3C")
            self.audio_frames = []
            self.sample_rate = 16000
            def callback(indata, frames, time, status):
                if self.recording_state:
                    self.audio_frames.append(indata.copy())
            self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", callback=callback)
            self.stream.start()
        else:
            self.recording_state = False
            self.voice_button.config(text="Speak", bg="#4CAF50")
            self.stream.stop()
            self.stream.close()
            audio = np.concatenate(self.audio_frames, axis=0)
            audio = np.squeeze(audio)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("small", device=device)
            result = model.transcribe(audio, fp16=torch.cuda.is_available())
            text = result["text"].strip()
            if text:
                self.user_input.delete("1.0", tk.END)
                self.user_input.insert(tk.END, text)
            else:
                messagebox.showinfo("Voice Input", "No speech detected. Please try again.")

    def clean_text(self, text):
        """Normalize text for speech synthesis by removing control characters and markdown markers."""
        cleaned = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ").replace("<", " ").replace(">", " ")
        cleaned = re.sub(r'(\*{1,2}|_{1,2})', '', cleaned)
        cleaned = " ".join(cleaned.split())
        return cleaned

    def split_text_by_punctuation(self, text):
        """Split text into sentences for chunked TTS playback."""
        parts = re.split(r'(?<=[\.\,\:\;\?\!])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def speak_text(self, text):
        """Convert text to speech in small chunks and play it.

        :param text: Text to voice.
        :type text: str
        """
        self.flag_speaking = True
        text = self.clean_text(text)
        chunks = self.split_text_by_punctuation(text)
        try:
            for chunk in chunks:
                tts = gTTS(chunk, lang="en")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    temp_file = fp.name
                    tts.save(temp_file)
                playsound.playsound(temp_file)
                os.remove(temp_file)
                time.sleep(0.05)
        except Exception as e:
            print("TTS Error:", e)

def interface():
    """Launch the GUI interface and return the final generated drives.

    :return: Final drives/missions output from UI.
    :rtype: str
    """
    init_conversation_file()
    root = tk.Tk()
    ui = ChatInterface(root)
    root.mainloop()
    results = ui.results
    print("Results..", results)
    return results

if __name__ == "__main__":
    final_answer = interface()
    print(final_answer)
    missions, drives = extract_missions_and_drives(final_answer)
    print("MISSIONS:\n", missions)
    print("\nDRIVES:\n", drives)