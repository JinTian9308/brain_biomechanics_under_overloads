// 全局变量声明
let mediaRecorder;
let audioChunks = [];
let isAudioInput = false; // 标志：是否为语音输入
const recordButton = document.getElementById('record-voice');
const chatInput = document.getElementById('chat-input');
const chatLog = document.getElementById('chat-log');
const recordingStatus = document.getElementById('recording-status');

// 初始化标志，防止多次调用
let chatInitialized = false;

// 初始化对话
async function initializeChat() {
    if (chatInitialized) return; // 避免重复初始化
    chatInitialized = true;

    try {
        const response = await fetch('/init-chat');
        if (!response.ok) throw new Error('Failed to fetch initial chat message');

        const data = await response.json();

        // 使用 appendMessage 插入初始消息，确保 "AI:" 加粗
        appendMessage('AI', data.message);
        speakText(data.message); // 前端播报消息
    } catch (error) {
        console.error('Failed to initialize chat:', error);
        appendMessage('AI', "Failed to load the AI assistant. Please refresh the page.");
    }
}

// 添加事件监听，确保初始化只触发一次
document.addEventListener('DOMContentLoaded', initializeChat);

// 开始录音
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        isAudioInput = true; // 标志为语音输入

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = handleAudioStop;

        mediaRecorder.start();
        recordingStatus.textContent = "Recording...";
    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Failed to access microphone. Please check your permissions.');
    }
}

// 停止录音
async function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        recordingStatus.textContent = "Processing...";
    }
}

// 处理音频停止
async function handleAudioStop() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('audio', audioBlob);

    try {
        // 显示“用户语音”在对话框中
        appendMessage('You', "Processing audio...");

        const response = await fetch('/process-input', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error('Error from backend:', errorData.error || 'Unknown error');
            alert('Failed to process audio: ' + (errorData.error || 'Unknown error'));
            return;
        }

        const data = await response.json();
        processBackendResponse(data);
    } catch (error) {
        console.error('Error during audio processing:', error);
        alert('Failed to transcribe audio.');
    } finally {
        recordingStatus.textContent = "";
        isAudioInput = false; // 重置标志
    }
}

// 处理发送文本消息
async function sendMessage() {
    const inputText = chatInput.value.trim();
    if (!inputText) {
        alert('Please enter a valid input.');
        return;
    }

    // 立即显示用户输入
    appendMessage('You', inputText);
    isAudioInput = false; // 标志为文本输入

    try {
        const response = await fetch('/process-input', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) throw new Error('Failed to process text');

        const data = await response.json();
        processBackendResponse(data);
    } catch (error) {
        console.error('Error during text processing:', error);
        alert('Failed to process your message. Please try again.');
    }
    chatInput.value = '';
}

// 处理后端返回的结果
function processBackendResponse(data) {
    if (data.transcription && isAudioInput) {
        // 如果是语音输入的转录内容，替换之前的“Processing audio...”文本
        const lastMessage = chatLog.lastElementChild;
        if (lastMessage && lastMessage.textContent === "You: Processing audio...") {
            lastMessage.remove();
        }
        appendMessage('You', data.transcription);
    }

    if (data.calculation_results) {
        // 在聊天框中显示简洁提示
        appendMessage('AI', data.chat_message);
        speakText(data.chat_message); // 语音播报

        // 在右上角显示详细计算结果
        const calculationResults = document.getElementById('calculation-results');
        calculationResults.innerHTML = `
            <p><strong>Maximum von Mises Stress:</strong> ${data.calculation_results["Maximum von Mises Stress"]}</p>
            <p><strong>Maximum Displacement:</strong> ${data.calculation_results["Maximum Displacement"]}</p>
            <p><strong>Maximum Pore Pressure:</strong> ${data.calculation_results["Maximum Pore Pressure"]}</p>
        `;

        // 显示图表
        const iframe = document.getElementById('chart-frame');
        iframe.srcdoc = data.chart_html;
    } else if (data.reply) {
        // 针对闲聊或错误的响应
        appendMessage('AI', data.reply);
        speakText(data.reply); // 语音播报
    }
}

// 查询特定点值
async function queryPoint() {
    const x = parseFloat(document.getElementById('x-coord').value);
    const y = parseFloat(document.getElementById('y-coord').value);
    const z = parseFloat(document.getElementById('z-coord').value);

    if (isNaN(x) || isNaN(y) || isNaN(z)) {
        alert('Please enter valid X, Y, Z coordinates.');
        return;
    }

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x, y, z }),
        });

        if (!response.ok) throw new Error('Failed to query point');

        const data = await response.json();
        alert(`Mises: ${data.Mises.toFixed(2)} kPa\nDisplacement: ${data.Displacement.toFixed(2)} mm\nPore Pressure: ${data.PorePressure.toFixed(2)} kPa`);
    } catch (error) {
        console.error('Error during query:', error);
        alert('Error querying point: ' + error.message);
    }
}

// 添加消息到聊天日志
function appendMessage(sender, message) {
    const formattedMessage = `<strong>${sender}:</strong> ${message}`;
    const lastMessage = chatLog.lastElementChild;

    // 如果最后一条消息和即将插入的消息相同，避免重复插入
    if (lastMessage && lastMessage.innerHTML === formattedMessage) {
        return;
    }

    chatLog.innerHTML += `<p>${formattedMessage}</p>`;
    chatLog.scrollTop = chatLog.scrollHeight; // 自动滚动到最新消息
}

// 语音播报
function speakText(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1; // 调整语速
    utterance.lang = 'en-US'; // 强制英语语音
    window.speechSynthesis.speak(utterance);
}

// 添加事件监听
recordButton.addEventListener('mousedown', startRecording);
recordButton.addEventListener('mouseup', stopRecording);
document.getElementById('send-message').addEventListener('click', sendMessage);
document.getElementById('query-point').addEventListener('click', queryPoint);
document.getElementById('download-results').addEventListener('click', async () => {
    try {
        const response = await fetch('/download-results');
        if (!response.ok) throw new Error('Failed to download results');

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = 'prediction_results.csv'; // 确保文件名正确
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error downloading file:', error);
        alert('Failed to download the file.');
    }
});
