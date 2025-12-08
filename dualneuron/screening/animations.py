import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
import os
import torch


def image_to_base64(tensor, quality=95, max_size=None):
    """Convert tensor (C, H, W) to base64 string.
    
    Args:
        tensor: Image tensor
        quality: JPEG quality (1-95). Use 95 for PNG-like quality, lower for smaller files.
        max_size: If set, resize image to this max dimension
    """
    if torch.is_tensor(tensor):
        img = tensor.cpu().numpy()
    else:
        img = tensor
    
    # Handle channel dimension
    if img.ndim == 3:
        if img.shape[0] in [1, 3, 4]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = img.squeeze(2)
    
    # Normalize to 0-255
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    img = img.astype(np.uint8)
    
    # Convert to PIL (always RGB for compatibility)
    if img.ndim == 2:
        pil_img = Image.fromarray(img, mode='L').convert('RGB')
    else:
        pil_img = Image.fromarray(img, mode='RGB')
    
    # Resize if needed
    if max_size is not None:
        w, h = pil_img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    
    # Encode
    buffer = BytesIO()
    if quality < 95:
        pil_img.save(buffer, format='JPEG', quality=quality, optimize=True)
        fmt = 'jpeg'
    else:
        pil_img.save(buffer, format='PNG', optimize=True)
        fmt = 'png'
    
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/{fmt};base64,{b64}"


def generate_interactive_viewer(
    neuron_id,
    dset,
    resp_dir,
    idx_dir,
    num_samples=100,
    savename='neuron_viewer.html',
    display_in_notebook=False,
    notebook_quality=50,
):
    """
    Create an interactive HTML viewer with slider for neuron activation visualization.
    """
    print("Loading data...")
    sorted_responses = np.load(os.path.join(resp_dir, f"{neuron_id}.npy")).astype(np.float64)
    sorted_dataset_indices = np.load(os.path.join(idx_dir, f"{neuron_id}.npy"))
    
    # Convert to Hz (multiply by 10)
    sorted_responses = sorted_responses * 10.0
    print(f"Response range: {sorted_responses.min():.2f} to {sorted_responses.max():.2f} Hz")
    
    # Adaptive sampling based on activation differences
    rng = np.random.default_rng(seed=num_samples)
    diffs = np.abs(np.diff(sorted_responses))
    probs = diffs / np.sum(diffs)
    
    sampled_transitions = rng.choice(
        len(probs),
        num_samples,
        p=probs,
        replace=False
    )
    sampled_positions = np.sort(sampled_transitions + 1)
    sampled_dataset_idx = sorted_dataset_indices[sampled_positions]
    sampled_activations = sorted_responses[sampled_positions]
    
    # Downsample the curve for plotting (max 1000 points)
    n_total = len(sorted_responses)
    max_curve_points = 1000
    if n_total > max_curve_points:
        curve_indices = np.linspace(0, n_total - 1, max_curve_points, dtype=int)
        curve_responses = sorted_responses[curve_indices].tolist()
        curve_positions = curve_indices.tolist()
    else:
        curve_responses = sorted_responses.tolist()
        curve_positions = list(range(n_total))
    
    print(f"Processing {num_samples} images...")
    images_b64 = []
    
    # Lower quality for notebook display
    if display_in_notebook:
        img_quality = notebook_quality
        img_max_size = 150
    else:
        img_quality = 95
        img_max_size = None
    
    for i, idx in enumerate(sampled_dataset_idx):
        tensor, _ = dset[idx]
        images_b64.append(image_to_base64(tensor, quality=img_quality, max_size=img_max_size))
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_samples} images...")
    
    # Prepare data for JavaScript
    data = {
        'neuronId': str(neuron_id),
        'numSamples': num_samples,
        'totalImages': n_total,
        'curveResponses': curve_responses,
        'curvePositions': curve_positions,
        'sampledPositions': sampled_positions.tolist(),
        'sampledActivations': sampled_activations.tolist(),
        'images': images_b64
    }
    
    html = generate_html(data)
    
    # if display_in_notebook:
    #     from IPython.display import HTML, display
    #     display(HTML(html))
    # else:
    #     with open(savename, 'w') as f:
    #         f.write(html)
    #     print(f"Done! Saved to {savename}")
    #     return html
    
    with open(savename, 'w') as f:
        f.write(html)
    print(f"Saved to {savename}")

    if display_in_notebook:
        from IPython.display import display
        from IPython.display import IFrame
        # Display the file in an IFrame. 
        # width='100%' uses available width, height needs to accommodate your UI (approx 800px)
        display(IFrame(src=savename, width='100%', height='850'))
        
        
def generate_interactive_viewer(
    neuron_id,
    dset,
    resp_dir,
    idx_dir,
    num_samples=100,
    savename='neuron_viewer.html',
    display_in_notebook=False,
    notebook_quality=50,
):
    """
    Create an interactive HTML viewer with slider for neuron activation visualization.
    """
    print("Loading data...")
    sorted_responses = np.load(os.path.join(resp_dir, f"{neuron_id}.npy")).astype(np.float64)
    sorted_dataset_indices = np.load(os.path.join(idx_dir, f"{neuron_id}.npy"))
    
    # Convert to Hz (multiply by 10)
    sorted_responses = sorted_responses * 10.0
    
    # Adaptive sampling
    rng = np.random.default_rng(seed=num_samples)
    diffs = np.abs(np.diff(sorted_responses))
    probs = diffs / np.sum(diffs)
    
    sampled_transitions = rng.choice(len(probs), num_samples, p=probs, replace=False)
    sampled_positions = np.sort(sampled_transitions + 1)
    sampled_dataset_idx = sorted_dataset_indices[sampled_positions]
    sampled_activations = sorted_responses[sampled_positions]
    
    # Curve downsampling
    n_total = len(sorted_responses)
    max_curve_points = 1000
    if n_total > max_curve_points:
        curve_indices = np.linspace(0, n_total - 1, max_curve_points, dtype=int)
        curve_responses = sorted_responses[curve_indices].tolist()
        curve_positions = curve_indices.tolist()
    else:
        curve_responses = sorted_responses.tolist()
        curve_positions = list(range(n_total))
    
    print(f"Processing {num_samples} images...")
    images_b64 = []
    
    # If displaying in notebook, we force lower quality/size to prevent
    # hitting the notebook's data rate limit with the Data URI.
    if display_in_notebook:
        img_quality = notebook_quality
        img_max_size = 150
    else:
        img_quality = 95
        img_max_size = None
    
    for i, idx in enumerate(sampled_dataset_idx):
        tensor, _ = dset[idx]
        images_b64.append(image_to_base64(tensor, quality=img_quality, max_size=img_max_size))
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_samples} images...")
    
    data = {
        'neuronId': str(neuron_id),
        'numSamples': num_samples,
        'totalImages': n_total,
        'curveResponses': curve_responses,
        'curvePositions': curve_positions,
        'sampledPositions': sampled_positions.tolist(),
        'sampledActivations': sampled_activations.tolist(),
        'images': images_b64
    }
    
    html = generate_html(data)

    if display_in_notebook:
        import base64
        from IPython.display import IFrame, display
        
        # Encode the HTML string to base64
        b64_html = base64.b64encode(html.encode('utf-8')).decode('utf-8')
        
        # Create a Data URI
        data_uri = f"data:text/html;base64,{b64_html}"
        
        # Display directly from memory - NO file is created
        print("Displaying in notebook (no file saved)...")
        display(IFrame(src=data_uri, width='100%', height='850'))
        
    else:
        # Standard behavior: Save to disk
        with open(savename, 'w') as f:
            f.write(html)
        print(f"Done! Saved to {savename}")
        

def generate_html(data):
    """Generate the complete HTML with embedded data and viewer."""
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neuron {neuronId} Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            background: #000;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 30px;
        }}
        
        h1 {{
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 30px;
            color: #fff;
        }}
        
        .container {{
            display: flex;
            gap: 50px;
            align-items: center;
            margin-bottom: 40px;
        }}
        
        .image-panel {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 18px;
        }}
        
        .image-container {{
            background: #0a0a0a;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            padding: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 320px;
            height: 320px;
        }}
        
        .image-container img {{
            max-width: 296px;
            max-height: 296px;
            image-rendering: -webkit-optimize-contrast;
        }}
        
        .sample-label {{
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.6);
            font-weight: 500;
        }}
        
        .activation-badge {{
            padding: 10px 20px;
            border-radius: 25px;
            background: #0a0a0a;
            border: 2px solid;
            font-weight: 700;
            font-size: 1.1rem;
            transition: all 0.15s ease;
        }}
        
        .plot-container {{
            background: #0a0a0a;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            padding: 8px;
        }}
        
        canvas {{ display: block; }}
        
        .controls {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
            width: 100%;
            max-width: 700px;
        }}
        
        .slider-container {{
            width: 100%;
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        input[type="range"] {{
            -webkit-appearance: none;
            flex: 1;
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(to right, #00d4ff, #ff0080);
            outline: none;
        }}
        
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: #fff;
            cursor: pointer;
            box-shadow: 0 0 12px rgba(255, 255, 255, 0.5);
            transition: transform 0.1s ease;
        }}
        
        input[type="range"]::-webkit-slider-thumb:hover {{
            transform: scale(1.15);
        }}
        
        input[type="range"]::-moz-range-thumb {{
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: #fff;
            cursor: pointer;
            border: none;
            box-shadow: 0 0 12px rgba(255, 255, 255, 0.5);
        }}
        
        .frame-display {{
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.5);
            min-width: 50px;
            text-align: center;
        }}
        
        .play-btn {{
            background: #0a0a0a;
            border: 2px solid rgba(255, 255, 255, 0.4);
            color: #fff;
            padding: 12px 35px;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .play-btn:hover {{
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.6);
        }}
    </style>
</head>
<body>
    <h1>Neuron {neuronId}</h1>
    
    <div class="container">
        <div class="image-panel">
            <div class="sample-label">Sample <span id="sampleNum">1</span> / {numSamples}</div>
            <div class="image-container">
                <img id="currentImage" src="" alt="Sample image">
            </div>
            <div id="activationBadge" class="activation-badge">0.0000 Hz</div>
        </div>
        
        <div class="plot-container">
            <canvas id="plotCanvas" width="600" height="400"></canvas>
        </div>
    </div>
    
    <div class="controls">
        <div class="slider-container">
            <span class="frame-display">1</span>
            <input type="range" id="frameSlider" min="0" max="{maxSlider}" value="0">
            <span class="frame-display">{numSamples}</span>
        </div>
        <button id="playBtn" class="play-btn">▶ Play</button>
    </div>

    <script>
        const DATA = {dataJson};
        
        let currentFrame = 0;
        let isPlaying = false;
        let playInterval = null;
        
        const slider = document.getElementById('frameSlider');
        const playBtn = document.getElementById('playBtn');
        const sampleNum = document.getElementById('sampleNum');
        const currentImage = document.getElementById('currentImage');
        const activationBadge = document.getElementById('activationBadge');
        const plotCanvas = document.getElementById('plotCanvas');
        const ctx = plotCanvas.getContext('2d');
        
        // Get global min/max for consistent color scaling
        const globalMinY = 0;
        const globalMaxY = Math.max.apply(null, DATA.curveResponses);
        const globalRangeY = globalMaxY - globalMinY || 1;
        
        // Color interpolation: cyan (#00d4ff) to magenta (#ff0080)
        function getColor(value) {{
            const t = (value - globalMinY) / globalRangeY;
            const r = Math.round(0 + t * 255);
            const g = Math.round(212 - t * 212);
            const b = Math.round(255 - t * 127);
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}
        
        function updateDisplay(frame) {{
            currentFrame = frame;
            slider.value = frame;
            sampleNum.textContent = frame + 1;
            
            currentImage.src = DATA.images[frame];
            
            const activation = DATA.sampledActivations[frame];
            const color = getColor(activation);
            activationBadge.textContent = activation.toFixed(2) + ' Hz';
            activationBadge.style.borderColor = color;
            activationBadge.style.color = color;
            
            drawPlot(frame);
        }}
        
        function drawPlot(frame) {{
            const w = plotCanvas.width, h = plotCanvas.height;
            const pad = {{ top: 30, right: 30, bottom: 55, left: 75 }};
            const plotW = w - pad.left - pad.right;
            const plotH = h - pad.top - pad.bottom;
            
            // Clear canvas
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, w, h);
            
            const curveX = DATA.curvePositions;
            const curveY = DATA.curveResponses;
            const totalImages = DATA.totalImages;
            
            const minY = globalMinY;
            const maxY = globalMaxY;
            const rangeY = globalRangeY;
            
            // Scale functions
            function scaleX(i) {{ return pad.left + (i / (totalImages - 1)) * plotW; }}
            function scaleY(v) {{ return pad.top + plotH - ((v - minY) / rangeY) * plotH; }}
            
            // Draw grid
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            for (let i = 0; i <= 4; i++) {{
                const y = pad.top + (plotH / 4) * i;
                ctx.beginPath();
                ctx.moveTo(pad.left, y);
                ctx.lineTo(w - pad.right, y);
                ctx.stroke();
            }}
            ctx.setLineDash([]);
            
            // Draw curve line with gradient color
            for (let i = 1; i < curveX.length; i++) {{
                ctx.beginPath();
                ctx.moveTo(scaleX(curveX[i-1]), scaleY(curveY[i-1]));
                ctx.lineTo(scaleX(curveX[i]), scaleY(curveY[i]));
                ctx.strokeStyle = getColor((curveY[i-1] + curveY[i]) / 2);
                ctx.lineWidth = 2.5;
                ctx.stroke();
            }}
            
            // Draw all sample points (colored by activation)
            for (let i = 0; i < DATA.sampledPositions.length; i++) {{
                ctx.beginPath();
                ctx.arc(scaleX(DATA.sampledPositions[i]), scaleY(DATA.sampledActivations[i]), 5, 0, Math.PI * 2);
                ctx.fillStyle = getColor(DATA.sampledActivations[i]);
                ctx.globalAlpha = 0.4;
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }}
            
            // Draw visited points (brighter)
            for (let i = 0; i < frame; i++) {{
                ctx.beginPath();
                ctx.arc(scaleX(DATA.sampledPositions[i]), scaleY(DATA.sampledActivations[i]), 6, 0, Math.PI * 2);
                ctx.fillStyle = getColor(DATA.sampledActivations[i]);
                ctx.globalAlpha = 0.85;
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }}
            
            // Draw current point with glow
            const currPosX = scaleX(DATA.sampledPositions[frame]);
            const currPosY = scaleY(DATA.sampledActivations[frame]);
            const currColor = getColor(DATA.sampledActivations[frame]);
            
            // Glow
            const glow = ctx.createRadialGradient(currPosX, currPosY, 0, currPosX, currPosY, 28);
            glow.addColorStop(0, currColor);
            glow.addColorStop(0.4, currColor.replace('rgb', 'rgba').replace(')', ', 0.4)'));
            glow.addColorStop(1, 'transparent');
            ctx.fillStyle = glow;
            ctx.beginPath();
            ctx.arc(currPosX, currPosY, 28, 0, Math.PI * 2);
            ctx.fill();
            
            // Current point
            ctx.beginPath();
            ctx.arc(currPosX, currPosY, 9, 0, Math.PI * 2);
            ctx.fillStyle = currColor;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2.5;
            ctx.stroke();
            
            // Draw axes
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(pad.left, pad.top);
            ctx.lineTo(pad.left, h - pad.bottom);
            ctx.lineTo(w - pad.right, h - pad.bottom);
            ctx.stroke();
            
            // X-axis label
            ctx.fillStyle = '#fff';
            ctx.font = '600 14px -apple-system, BlinkMacSystemFont, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Sorted Image Index', pad.left + plotW / 2, h - 15);
            
            // Y-axis label
            ctx.save();
            ctx.translate(20, pad.top + plotH / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Activation (Hz)', 0, 0);
            ctx.restore();
            
            // Y-axis tick labels
            ctx.font = '500 12px -apple-system, BlinkMacSystemFont, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            for (let i = 0; i <= 4; i++) {{
                const val = maxY - (rangeY / 4) * i;
                const y = pad.top + (plotH / 4) * i;
                ctx.fillText(val.toFixed(1), pad.left - 10, y + 4);
            }}
            
            // X-axis tick labels
            ctx.textAlign = 'center';
            const xTicks = [0, 0.25, 0.5, 0.75, 1];
            for (let t of xTicks) {{
                const idx = Math.round(t * (totalImages - 1));
                const x = scaleX(idx);
                ctx.fillText(idx.toLocaleString(), x, h - pad.bottom + 18);
            }}
        }}
        
        slider.addEventListener('input', (e) => updateDisplay(parseInt(e.target.value)));
        
        playBtn.addEventListener('click', () => {{
            isPlaying = !isPlaying;
            if (isPlaying) {{
                playBtn.textContent = '⏸ Pause';
                playInterval = setInterval(() => {{
                    let next = currentFrame + 1;
                    if (next >= DATA.numSamples) next = 0;
                    updateDisplay(next);
                }}, 200);
            }} else {{
                playBtn.textContent = '▶ Play';
                clearInterval(playInterval);
            }}
        }});
        
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft' && currentFrame > 0) updateDisplay(currentFrame - 1);
            else if (e.key === 'ArrowRight' && currentFrame < DATA.numSamples - 1) updateDisplay(currentFrame + 1);
            else if (e.key === ' ') {{ e.preventDefault(); playBtn.click(); }}
        }});
        
        updateDisplay(0);
    </script>
</body>
</html>'''
    
    return html_template.format(
        neuronId=data['neuronId'],
        numSamples=data['numSamples'],
        maxSlider=data['numSamples'] - 1,
        dataJson=json.dumps(data)
    )