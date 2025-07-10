const CACHE_NAME = 'face-recognition-app-v1';
const urlsToCache = [
  '/',
  '/static/css/style.css',
  '/static/js/app.js',
  '/static/icon-192x192.png',
  '/static/icon-512x512.png',
  '/manifest.json'
];

// Service Worker インストール時
self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        console.log('Opened cache');
        return cache.addAll(urlsToCache);
      })
  );
});

// Service Worker 有効化時
self.addEventListener('activate', function(event) {
  event.waitUntil(
    caches.keys().then(function(cacheNames) {
      return Promise.all(
        cacheNames.map(function(cacheName) {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// フェッチイベント（リクエストの処理）
self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request)
      .then(function(response) {
        // キャッシュにある場合はそれを返す
        if (response) {
          return response;
        }

        // キャッシュにない場合はネットワークから取得
        return fetch(event.request).then(function(response) {
          // 無効なレスポンスの場合はそのまま返す
          if (!response || response.status !== 200 || response.type !== 'basic') {
            return response;
          }

          // レスポンスをクローンしてキャッシュに保存
          const responseToCache = response.clone();
          caches.open(CACHE_NAME)
            .then(function(cache) {
              cache.put(event.request, responseToCache);
            });

          return response;
        }).catch(function() {
          // ネットワークエラーの場合、オフラインページを表示
          if (event.request.destination === 'document') {
            return caches.match('/');
          }
        });
      })
  );
});

// バックグラウンド同期（オンライン復帰時の処理）
self.addEventListener('sync', function(event) {
  if (event.tag === 'background-sync') {
    event.waitUntil(
      // オンライン復帰時に実行したい処理をここに記述
      console.log('Background sync triggered')
    );
  }
});

// プッシュ通知（必要に応じて）
self.addEventListener('push', function(event) {
  const options = {
    body: event.data ? event.data.text() : 'Default message',
    icon: '/static/icon-192x192.png',
    badge: '/static/icon-192x192.png'
  };

  event.waitUntil(
    self.registration.showNotification('Face Recognition App', options)
  );
});