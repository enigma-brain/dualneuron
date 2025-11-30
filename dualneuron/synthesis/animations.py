import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
import torch


def image_to_base64(tensor, quality=95, max_size=None):
    """Convert tensor (C, H, W) or (H, W, C) to base64 string."""
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


def generate_optimization_viewer(
    images,
    activations,
    savename='optimization_viewer.html',
    display_in_notebook=False,
    notebook_quality=50,
    title="Optimization Progress"
):
    """
    Create an interactive HTML viewer for optimization sequence.
    
    Args:
        images: List of images (tensors or numpy arrays) showing optimization progress
        activations: List or array of activation values (will be converted to Hz by *10)
        savename: Output HTML filename
        display_in_notebook: If True, display inline in Jupyter
        notebook_quality: Image quality for notebook display (1-95)
        title: Title for the viewer
    """
    print("Processing optimization sequence...")
    
    # Convert activations to Hz
    activations = np.array(activations) * 10.0
    num_steps = len(images)
    
    print(f"Number of steps: {num_steps}")
    print(f"Activation range: {activations.min():.2f} to {activations.max():.2f} Hz")
    
    # Process images
    images_b64 = []
    
    if display_in_notebook:
        img_quality = notebook_quality
        img_max_size = 150
    else:
        img_quality = 95
        img_max_size = None
    
    print(f"Encoding {num_steps} images...")
    for i, img in enumerate(images):
        images_b64.append(image_to_base64(img, quality=img_quality, max_size=img_max_size))
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_steps} images...")
    
    # Prepare data for JavaScript
    data = {
        'title': title,
        'numSteps': num_steps,
        'activations': activations.tolist(),
        'images': images_b64
    }
    
    html = generate_html(data)
    
    if display_in_notebook:
        import base64 as b64_module
        from IPython.display import IFrame, display
        
        # Encode the HTML string to base64
        b64_html = b64_module.b64encode(html.encode('utf-8')).decode('utf-8')
        
        # Create a Data URI
        data_uri = f"data:text/html;base64,{b64_html}"
        
        # Display directly from memory
        print("Displaying in notebook (no file saved)...")
        display(IFrame(src=data_uri, width='100%', height='850'))
        
    else:
        # Save to disk
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
    <title>{title}</title>
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
            padding: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 320px;
            height: 320px;
        }}
        
        .image-container img {{
            width: 304px;
            height: 304px;
            object-fit: contain;
            image-rendering: -webkit-optimize-contrast;
        }}
        
        .step-label {{
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
    <h1>{title}</h1>
    
    <div class="container">
        <div class="image-panel">
            <div class="step-label">Step <span id="stepNum">0</span> / {maxStep}</div>
            <div class="image-container">
                <img id="currentImage" src="" alt="Optimization step">
            </div>
            <div id="activationBadge" class="activation-badge">0.0000 Hz</div>
        </div>
        
        <div class="plot-container">
            <canvas id="plotCanvas" width="600" height="400"></canvas>
        </div>
    </div>
    
    <div class="controls">
        <div class="slider-container">
            <span class="frame-display">0</span>
            <input type="range" id="frameSlider" min="0" max="{maxSlider}" value="0">
            <span class="frame-display">{maxStep}</span>
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
        const stepNum = document.getElementById('stepNum');
        const currentImage = document.getElementById('currentImage');
        const activationBadge = document.getElementById('activationBadge');
        const plotCanvas = document.getElementById('plotCanvas');
        const ctx = plotCanvas.getContext('2d');
        
        // Get global min/max for consistent color scaling
        const activations = DATA.activations;
        const globalMinY = Math.min.apply(null, activations);
        const globalMaxY = Math.max.apply(null, activations);
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
            stepNum.textContent = frame;
            
            currentImage.src = DATA.images[frame];
            
            const activation = activations[frame];
            const color = getColor(activation);
            activationBadge.textContent = activation.toFixed(4) + ' Hz';
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
            
            const numSteps = DATA.numSteps;
            const minY = globalMinY;
            const maxY = globalMaxY;
            const rangeY = globalRangeY;
            
            // Scale functions
            function scaleX(step) {{ return pad.left + (step / (numSteps - 1)) * plotW; }}
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
            
            // Draw full trajectory with gradient color
            for (let i = 1; i < numSteps; i++) {{
                ctx.beginPath();
                ctx.moveTo(scaleX(i-1), scaleY(activations[i-1]));
                ctx.lineTo(scaleX(i), scaleY(activations[i]));
                ctx.strokeStyle = getColor((activations[i-1] + activations[i]) / 2);
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.4;
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            }}
            
            // Draw visited trajectory (brighter)
            for (let i = 1; i <= frame; i++) {{
                ctx.beginPath();
                ctx.moveTo(scaleX(i-1), scaleY(activations[i-1]));
                ctx.lineTo(scaleX(i), scaleY(activations[i]));
                ctx.strokeStyle = getColor((activations[i-1] + activations[i]) / 2);
                ctx.lineWidth = 3.5;
                ctx.globalAlpha = 0.85;
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            }}
            
            // Draw all points (faint)
            for (let i = 0; i < numSteps; i++) {{
                ctx.beginPath();
                ctx.arc(scaleX(i), scaleY(activations[i]), 4, 0, Math.PI * 2);
                ctx.fillStyle = getColor(activations[i]);
                ctx.globalAlpha = 0.3;
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }}
            
            // Draw visited points (brighter)
            for (let i = 0; i < frame; i++) {{
                ctx.beginPath();
                ctx.arc(scaleX(i), scaleY(activations[i]), 5, 0, Math.PI * 2);
                ctx.fillStyle = getColor(activations[i]);
                ctx.globalAlpha = 0.7;
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }}
            
            // Draw current point with glow
            const currPosX = scaleX(frame);
            const currPosY = scaleY(activations[frame]);
            const currColor = getColor(activations[frame]);
            
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
            ctx.fillText('Optimization Step', pad.left + plotW / 2, h - 15);
            
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
                ctx.fillText(val.toFixed(2), pad.left - 10, y + 4);
            }}
            
            // X-axis tick labels
            ctx.textAlign = 'center';
            const xTicks = [0, 0.25, 0.5, 0.75, 1];
            for (let t of xTicks) {{
                const step = Math.round(t * (numSteps - 1));
                const x = scaleX(step);
                ctx.fillText(step.toString(), x, h - pad.bottom + 18);
            }}
        }}
        
        slider.addEventListener('input', (e) => updateDisplay(parseInt(e.target.value)));
        
        playBtn.addEventListener('click', () => {{
            isPlaying = !isPlaying;
            if (isPlaying) {{
                playBtn.textContent = '⏸ Pause';
                playInterval = setInterval(() => {{
                    let next = currentFrame + 1;
                    if (next >= DATA.numSteps) next = 0;
                    updateDisplay(next);
                }}, 80);
            }} else {{
                playBtn.textContent = '▶ Play';
                clearInterval(playInterval);
            }}
        }});
        
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft' && currentFrame > 0) updateDisplay(currentFrame - 1);
            else if (e.key === 'ArrowRight' && currentFrame < DATA.numSteps - 1) updateDisplay(currentFrame + 1);
            else if (e.key === ' ') {{ e.preventDefault(); playBtn.click(); }}
        }});
        
        updateDisplay(0);
    </script>
</body>
</html>'''
    
    return html_template.format(
        title=data['title'],
        numSteps=data['numSteps'],
        maxStep=data['numSteps'] - 1,
        maxSlider=data['numSteps'] - 1,
        dataJson=json.dumps(data)
    )