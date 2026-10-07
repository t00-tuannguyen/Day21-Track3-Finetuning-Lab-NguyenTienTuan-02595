# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Cùng một cấu hình LoRA, cùng model, cùng số step: trên corpus mặc định nó thắng prompt tốt
nhất +0.205, trên bình luận thật (ViSFD) nó thua −0.050. Tôi đã nghĩ cấu hình "vùng không hối
tiếc" là thứ quyết định; hoá ra dữ liệu quyết định dấu của kết quả. Điều ngạc nhiên thứ hai là
`attn_only` — cấu hình mà deck gọi là Lỗi #1 — không thua `correct` ở cả hai lần đo.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Việc chạy train dữ liệu mất khá nhiều thời gian. Đây là chỗ tôi đã dự đoán trước vì quá trình train cần phải chạy từng bộ dữ liệu và chạy qua rất nhiều tham số.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng tin "fine-tune trên dữ liệu của mình thì chắc chắn tốt hơn prompt". Giờ tôi thấy nó
chỉ đúng khi dữ liệu đủ nhiều và đủ nhất quán để model học được thứ mà prompt không nói ra được.
Với 225 ví dụ, prompt có định nghĩa rõ ràng thắng — và rẻ hơn, nhanh hơn, không làm model quên
kiến thức chung (regression của bản fine-tune rơi từ 0.791 xuống 0.280).

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code để: đọc repo và lập kế hoạch; tìm dataset trên Hugging Face và viết script
dựng corpus ViSFD (lọc, khử nhiễm, lấy mẫu cố định); thêm scorer F1 theo khía cạnh và công tắc
task; điều khiển Colab qua trình duyệt; và viết nháp report này. Những chỗ nó sai hoặc tôi phải
can thiệp: (1) nó để file kết quả nằm trên Colab gần một tiếng thay vì tải về ngay, nên mất một
lần chạy; (2) khi máy ảo đã rơi về CPU, nó suýt chạy lại pipeline trên CPU trước khi phát hiện
không có GPU; (3) trong bản nháp report, nó viết vài câu mạnh hơn dữ liệu cho phép (ví dụ "phần
lớn lỗi của (b) là lỗi cực") và phải tự sửa lại sau khi tính lại số. Tôi kiểm tra mọi con số
trong report với file trong `results/`.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Viết prompt tốt nhất có thể cho bài toán của họ và đo nó trên một tập eval đóng băng — kèm một
tập câu hỏi phổ thông để đo hồi quy — **trước khi** train bất cứ thứ gì. Nếu prompt đã đủ tốt
thì không fine-tune. Nếu chưa, việc tiếp theo là xem dữ liệu của họ có đủ lớn và nhất quán
không, và chuẩn bị sẵn 1–5% dữ liệu replay ngay từ đầu.
