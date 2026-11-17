#pragma once

#include <string_view>

namespace neuron {
inline constexpr std::string_view web_chat = R"HTML(<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neuron chat</title>
<style>
  :root { color-scheme: dark; font: 16px/1.6 system-ui, sans-serif; background: #10151b; color: #e5eaf0; }
  body { max-width: 800px; margin: auto; padding: 24px; }
  header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #35404c; }
  h1 { font-size: 24px; letter-spacing: -.5px; margin: 0; }
  #status { color: #9aabba; font-size: 14px; }
  #messages { min-height: 45vh; margin: 24px 0; }
  article { margin: 16px 0; padding: 16px 20px; border-radius: 12px; background: #1b242e; overflow-wrap: anywhere; }
  article.user { background: #243a42; }
  article strong { display: block; font-size: 12px; color: #a0b4c4; text-transform: uppercase; letter-spacing: 1px; }
  article p { white-space: pre-wrap; margin: 6px 0 0; }
  textarea { width: 100%; box-sizing: border-box; resize: vertical; min-height: 90px; background: #19222c; color: inherit; border: 1px solid #465561; border-radius: 8px; padding: 12px; font: inherit; }
  button { border: 1px solid #56707c; border-radius: 7px; background: #253843; color: inherit; padding: 8px 16px; cursor: pointer; font: inherit; }
  button.primary { background: #b8e3c1; color: #102217; border-color: #b8e3c1; }
  button:disabled { opacity: .4; cursor: default; }
  .actions { display: flex; gap: 8px; margin-top: 10px; }
  #error { color: #ffb3a8; white-space: pre-wrap; }
  footer { color: #9aabba; font-size: 13px; margin-top: 16px; }
</style>
<header><div><h1>Neuron</h1><span id="status">Connecting...</span></div><button id="reset">Reset chat</button></header>
<main>
  <div id="messages" role="log" aria-label="Conversation"></div>
  <form id="form">
    <label for="prompt">Message</label>
    <textarea id="prompt" placeholder="Ask a question" required></textarea>
    <div class="actions"><button class="primary" id="send" disabled>Send</button><button type="button" id="stop" disabled>Stop reply</button></div>
  </form>
  <p id="error" role="alert"></p>
</main>
<footer>Runs on your CPU. Enter to send, Shift+Enter for a new line. Reset when the context is full.</footer>
<script>
  'use strict';
  const $ = id => document.getElementById(id);
  let model = '', history = [], controller = null;
  function message(role, text) {
    const article = document.createElement('article');
    article.className = role;
    const label = document.createElement('strong'), content = document.createElement('p');
    label.textContent = role === 'user' ? 'You' : 'Neuron';
    content.textContent = text;
    article.append(label, content); $('messages').append(article);
    return content;
  }
  async function connect() {
    try {
      const response = await fetch('/v1/models');
      if (!response.ok) throw new Error('Could not load models');
      model = (await response.json()).data[0].id;
      $('status').textContent = model;
      $('send').disabled = false;
    } catch (error) { $('error').textContent = error.message; }
  }
  $('reset').onclick = () => { history = []; $('messages').replaceChildren(); $('error').textContent = ''; $('prompt').focus(); };
  $('stop').onclick = () => controller?.abort();
  $('prompt').onkeydown = event => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); if (!controller) $('form').requestSubmit(); }
  };
  $('form').onsubmit = async event => {
    event.preventDefault();
    if (controller || !model || !$('prompt').value.trim()) return;
    const text = $('prompt').value;
    const previous = history.length;
    history.push({role: 'user', content: text});
    const userNode = message('user', text), answer = message('assistant', '');
    $('prompt').value = ''; $('error').textContent = '';
    controller = new AbortController();
    $('send').disabled = true; $('reset').disabled = true; $('stop').disabled = false;
    let done = false;
    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, signal: controller.signal,
        body: JSON.stringify({model, messages: history, max_tokens: 128, stream: true, temperature: 0})
      });
      if (!response.ok) { const body = await response.json(); throw new Error(body.error.message); }
      const reader = response.body.getReader(), decoder = new TextDecoder();
      let pending = '';
      while (!done) {
        const part = await reader.read();
        if (part.done) break;
        pending += decoder.decode(part.value, {stream: true});
        let end;
        while ((end = pending.indexOf('\n\n')) !== -1) {
          const frame = pending.slice(0, end); pending = pending.slice(end + 2);
          for (const line of frame.split('\n')) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6);
            if (data === '[DONE]') { done = true; break; }
            const chunk = JSON.parse(data);
            if (chunk.error) throw new Error(chunk.error.message);
            answer.textContent += chunk.choices[0]?.delta?.content || '';
          }
          window.scrollTo(0, document.body.scrollHeight);
        }
      }
      if (!done) throw new Error('The server closed the reply early.');
    } catch (error) {
      $('error').textContent = error.name === 'AbortError' ? 'Reply stopped.' : error.message;
    } finally {
      if (answer.textContent) history.push({role: 'assistant', content: answer.textContent});
      else { history.length = previous; answer.parentElement.remove(); userNode.parentElement.remove(); }
      controller = null; $('send').disabled = false; $('reset').disabled = false; $('stop').disabled = true;
      $('prompt').focus();
    }
  };
  connect();
</script>
</html>)HTML";
}  // namespace neuron
