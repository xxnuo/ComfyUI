# import uvicorn

# if __name__ == "__main__":
#     uvicorn.run("webui.server:app", host="0.0.0.0", port=3000)

import torch
from webui.engine import WanVideo

if __name__ == "__main__":
    with torch.no_grad():
        wanvideo = WanVideo()
        prompt = "A misty dawn unfolds in an ancient cherry-blossom grove as a sleek, silvery-furred cat in ornate samurai armor—with dark indigo and crimson plates, gold filigree, and cherry-blossom motifs—stands poised in a lone clearing surrounded by playful yet determined dog warriors in battered leather gear; in one fluid motion it draws a miniature katana (paw-shaped guard, red silk hilt) and wakizashi as its amber eyes lock onto its foes beneath a crimson headband, then darts forward in a graceful leap, low-tracking the camera alongside its lightning-fast parries and strikes, slow-motion flashes of steel catching pre-dawn light and scattering petals, the stylized paw-within-waves chest crest and swaying omamori charm hinting at its heritage, all underscored by soft wind, metallic clashes, distant growls, and a rising taiko drum swell, until it weaves between three attackers, halts center-frame, chest heaving with disciplined confidence, and the scene fades out on its silhouette against a cascade of blossoms."
        output_video_path = wanvideo.inference(prompt=prompt)
        print(f"Video saved to {output_video_path}")
