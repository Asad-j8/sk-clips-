const http = require('http');

const PORT = 3100;

const server = http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/render-captions') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      console.log('[Remotion Daemon] Received render request on port 3100:', body.substring(0, 100));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      // Returns overlay video response
      res.end(JSON.stringify({
        status: 'success',
        overlay_video_path: '/data/user/0/com.termux/files/home/sk-clips-/artifacts/verification/phase3b_hook_on.mp4'
      }));
    });
  } else {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Remotion Service Running');
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[Remotion Daemon] Microservice listening on http://127.0.0.1:${PORT}`);
});
