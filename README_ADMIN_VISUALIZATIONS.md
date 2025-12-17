# Admin Visualizations - Pre-generation Guide

## Vấn đề
Khi deploy trên Render, trang admin có thể bị **Internal Server Error** do:
- Plotly visualization tốn nhiều memory khi generate trực tiếp
- Render free tier có giới hạn RAM (~512MB-1GB)

## Giải pháp
**Pre-generate** các visualization HTML trước khi deploy, sau đó admin view chỉ cần đọc file HTML đã có sẵn.

## Cách sử dụng

### 1. Generate visualizations (chạy trước khi deploy)
```bash
python scripts/generate_admin_visualizations.py
```

Script này sẽ:
- Load ratings và movies data
- Sample data (max 100k rows) để giảm memory
- Generate 4 visualizations:
  - `rating_distribution.html`
  - `genre_frequency.html`
  - `top_movies.html`
  - `genre_rating_heatmap.html`
- Lưu vào `app/static/visualizations/`

### 2. Deploy
- Đảm bảo các file HTML đã được generate và commit vào git
- Khi deploy trên Render, admin view sẽ đọc từ các file này
- **Không cần** import plotly trong runtime → giảm memory footprint

### 3. Update visualizations
Nếu data thay đổi, chạy lại script:
```bash
python scripts/generate_admin_visualizations.py
```

## File structure
```
app/
  static/
    visualizations/          # Pre-generated HTML files
      rating_distribution.html
      genre_frequency.html
      top_movies.html
      genre_rating_heatmap.html
```

## Fallback
Nếu file HTML chưa có, admin view sẽ hiển thị thông báo yêu cầu chạy script generate.

## Lợi ích
✅ Giảm memory usage khi runtime  
✅ Tránh OOM trên Render  
✅ Load nhanh hơn (không cần tính toán mỗi request)  
✅ Có thể cache các file HTML trên CDN  

