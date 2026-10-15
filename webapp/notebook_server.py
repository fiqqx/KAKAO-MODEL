"""Server yang dapat dimulai dan dihentikan dari sel notebook VS Code."""

from threading import Thread


class NotebookServerManager:
    """Satu server per manager; start berulang tidak membuat server baru."""

    def __init__(self):
        self._server = None
        self._thread = None
        self._url = None

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, app, host="127.0.0.1", port=5000):
        if self.is_running:
            return self._url
        from waitress import create_server
        # create_server langsung mengikat port; bentrok port terlihat pada sel.
        self._server = create_server(
            app, host=host, port=int(port), threads=4, asyncore_loop_timeout=0.2,
        )
        effective_port = self._server.effective_port
        browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        self._url = f"http://{browser_host}:{effective_port}"
        self._thread = Thread(target=self._server.run, name="KakaoWebsite", daemon=True)
        self._thread.start()
        return self._url

    def stop(self):
        if self._server is not None:
            # Menutup listener saja meninggalkan worker Waitress. Tutup keduanya.
            from waitress import wasyncore
            self._server.task_dispatcher.shutdown(cancel_pending=True, timeout=3)
            self._server.close()
            wasyncore.close_all(map=self._server._map)
        if self._thread is not None:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                raise RuntimeError("Server belum berhenti. Tunggu inferensi aktif selesai, lalu klik Hentikan lagi.")
        self._server = None
        self._thread = None
        self._url = None
