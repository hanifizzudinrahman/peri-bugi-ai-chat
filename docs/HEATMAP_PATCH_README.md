# peri-bugi-ai-chat — Heatmap RnD Patch

## Files dalam ZIP ini

```
app/agents/nodes/generate.py   ← MODIFIED (Phase 6 forward image_artifacts)
```

## Cara Apply

```powershell
cd "C:\Hanif Digitech\Internal Projects\peri-bugi-ai-chat"

git stash

Expand-Archive -Path peri-bugi-ai-chat-heatmap-patch.zip -DestinationPath . -Force

docker compose restart ai-chat
docker compose logs ai-chat --tail 30
```

## Apa yang Berubah

Di Phase 6 `generate.py` (image_artifacts forward ke chat metadata), dictionary push diperluas:

**Sebelum:**
```python
artifacts_list.append({
    "view_type": view_type,
    "crop_image_url": crop_url,
    "overlay_image_url": overlay_url,
})
```

**Sesudah:**
```python
mask_overlay_url = artifacts.get("mask_overlay_image_url")
heatmap_url = artifacts.get("heatmap_image_url")

if crop_url or overlay_url or mask_overlay_url or heatmap_url:
    artifacts_list.append({
        "view_type": view_type,
        "crop_image_url": crop_url,
        "overlay_image_url": overlay_url,
        "mask_overlay_image_url": mask_overlay_url,
        "heatmap_image_url": heatmap_url,
    })
```

## Test

1. Pastikan ai-cv sudah patched dengan `OVERLAY_RENDER_MODE=both` (atau heatmap).
2. Trigger Tanya Peri image upload via `/tanya` (parent) atau `/admin/tanya-peri/sandbox`.
3. Cek SSE `done` event payload di browser DevTools Network → SSE message → metadata harus include `image_artifacts[].mask_overlay_image_url` dan `heatmap_image_url`.
4. Reload session → cek ChatMessage di DB → `metadata.image_artifacts[]` harus persist field baru.

## Backward Compat

- Mode `mask` di ai-cv → field baru null → forward null. FE handle null gracefully.
- Session lama (`metadata.image_artifacts` tanpa field baru) → no impact, karena patch ini cuma di write path. Read path (chat_service `_refresh_metadata_artifacts` di api) sudah defensive.

## Rollback

Revert `generate.py` ke version sebelum patch — dict push 3-field saja. Field baru di metadata yang sudah ke-write tetap aman di DB, cuma gak di-forward ke FE saat new message.
