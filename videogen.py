# import uvicorn

# if __name__ == "__main__":
#     uvicorn.run("webui.server:app", host="0.0.0.0", port=3000)

import torch
from webui.engine import WanVideo

if __name__ == "__main__":
    with torch.no_grad():
        wanvideo = WanVideo()
        # prompt = "A misty dawn unfolds in an ancient cherry-blossom grove as a sleek, silvery-furred cat in ornate samurai armor—with dark indigo and crimson plates, gold filigree, and cherry-blossom motifs—stands poised in a lone clearing surrounded by playful yet determined dog warriors in battered leather gear; in one fluid motion it draws a miniature katana (paw-shaped guard, red silk hilt) and wakizashi as its amber eyes lock onto its foes beneath a crimson headband, then darts forward in a graceful leap, low-tracking the camera alongside its lightning-fast parries and strikes, slow-motion flashes of steel catching pre-dawn light and scattering petals, the stylized paw-within-waves chest crest and swaying omamori charm hinting at its heritage, all underscored by soft wind, metallic clashes, distant growls, and a rising taiko drum swell, until it weaves between three attackers, halts center-frame, chest heaving with disciplined confidence, and the scene fades out on its silhouette against a cascade of blossoms."
        # prompt = 'A late-night soft glow spills across a messy, modern office as a fluffy, slightly chubby ginger cat in a wrinkled miniature suit vest and a loosened tie sits impatiently before a massive, glowing mousepad, hemmed in by menacing stacks of paperwork and relentlessly flashing red error pop-ups on a triple-monitor setup; with a deep sigh, it expertly adjusts a tiny Bluetooth earpiece with one paw, then in one fluid motion, it grips an ergonomic optical mouse (fishbone-scroll-wheel, soft-grip sides) while the other poises over a mini mechanical keyboard, its amber eyes locking onto the critical error code under the glare of a desk lamp, then plunges into its work, the camera low-tracking its lightning-fast clicks and drags, slow-motion flashes of keystrokes making the backlights flare in the gloom and dismissing virtual error reports like confetti, the embroidered company logo on its breast pocket (a stylized ball-of-yarn icon) and a swaying ID badge (bearing a stern-faced photo) hinting at its corporate station, all underscored by the low whir of computer fans, crisp keyboard clicks, incessant system alert pings, and a rising lo-fi beat, until it finally isolates a target in a complex interface, hovers the cursor over the "Execute" button, brings its full body weight down in a decisive pounce, lets out a satisfied yawn, and the scene fades to black on its silhouette illuminated by the green "Task Complete" notification.'
        prompt = """
赛博朋克CG数字艺术。在霓虹璀璨的未来都市雨夜，一只矫健的黑色忍者小猫蹲踞在高楼边缘。它系着一条红色面巾，遮住口鼻，碧绿的眼眸锐利地凝视着下方。雨水打湿了它的皮毛和脚下的金属屋顶，反射着城市闪烁的全息广告牌和穿梭飞行器的流光。中景，略微仰视的视角。
"""
        output_video_path = wanvideo.inference(prompt=prompt, shift=12.0)
        print(f"Video saved to {output_video_path}")
