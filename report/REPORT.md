# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Python 3.13.15 / PyTorch 2.11.0+cu128 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): `class_id: 468`, `class_name: "cab"`, `rank: 1`, `score: 0.510915`, `taxonomy_name: "ImageNet-1K"`.
- Record này mô tả toàn ảnh như thế nào? Record này gán một nhãn phân loại duy nhất cho toàn bộ bức ảnh (image-level prediction), đại diện cho chủ thể/nội dung tổng thể nổi bật nhất của khung hình mà không khoanh vùng vị trí hay tách biệt các thực thể riêng rẽ.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Danh mục 1.000 lớp được định nghĩa bởi bộ dữ liệu chuẩn ImageNet-1K, đã được cố định trong quá trình tiền huấn luyện model `yolo11n-cls.pt`.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? Giữ cả 3 trường giúp: (1) ID hỗ trợ truy xuất và xử lý máy tối ưu, tránh lỗi ký tự; (2) Tên lớp giúp con người/annotator đọc hiểu trực quan; (3) Tên taxonomy định danh chính xác không gian nhãn, ngăn ngừa xung đột ngữ nghĩa khi map giữa các tập dữ liệu khác nhau (ví dụ ImageNet vs COCO).
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline cần nêu rõ quy tắc ưu tiên chọn nhãn: chọn chủ thể chiếm diện tích lớn nhất, nằm ở trung tâm/tiền cảnh, hoặc quy định chuyển đổi bài toán sang phân loại đa nhãn (multi-label) / phát hiện vật thể (object detection) nếu cần bao quát mọi đối tượng.
- Vì sao model score không phải ground truth? Model score chỉ là giá trị xác suất phân phối thống kê (confidence score) do mạng nơ-ron sinh ra dựa trên trọng số đã học, có thể gặp lỗi phân loại sai hoặc overconfidence; trong khi ground truth là nhãn chân lý được con người kiểm chứng và xác nhận theo guideline chuẩn.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): `class_name: "person"`, `score: 0.912625`, `bbox_xyxy: [385.33, 69.24, 498.92, 348.92]`, `bbox_width: 113.58`, `bbox_height: 279.68`.
- Diễn giải vị trí box bằng lời: Hộp bao (`bounding box`) của người (`person`) nằm ở vùng bên phải - trung tâm của bức ảnh, có góc trên-trái tại toạ độ pixel (x=385.33, y=69.24) và góc dưới-phải tại (x=498.92, y=348.92), với chiều rộng 113.58 pixel và chiều cao 279.68 pixel, bao trọn vóc dáng người đang đứng trong căn bếp.
- So sánh số prediction ở hai threshold: Khi đặt threshold cao (0.35), mô hình chỉ giữ lại các đối tượng có độ tin cậy rõ nét (ví dụ 9 predictions cho sample kitchen gồm person, bowl, oven, cup). Khi hạ threshold (ví dụ 0.15 hoặc 0.25), số lượng prediction tăng lên đáng kể do giữ lại thêm nhiều dự đoán điểm số thấp và đối tượng nhỏ/mờ.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Hạ threshold giúp tăng độ bao phủ (recall), giảm nguy cơ bỏ sót vật thể nhưng làm tăng tỷ lệ dương tính giả (false positives), khiến khối lượng công việc của reviewer/QC tăng mạnh vì phải lọc và xóa nhiều box rác. Nâng threshold giảm bớt việc cho reviewer nhưng dễ làm sót các đối tượng quan trọng (false negatives).
- Đề xuất một quy tắc box chặt: Bounding box phải ôm sát các điểm cực viền ngoài cùng của vật thể theo pixel, không để khoảng trống thừa vượt quá 2-3 pixel và tuyệt đối không cắt lẹm vào ranh giới thực thể.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline/escalation cần quy định rõ: (1) Ngưỡng tỷ lệ che khuất tối đa cho phép gán nhãn (ví dụ nếu bị che >80% và mất đặc trưng nhận diện thì bỏ qua); (2) Quy định vẽ box chỉ bao phần nhìn thấy (visible box) hay vẽ ước lượng toàn bộ hình thể (amodal box); (3) Đối với vật thể bị cắt mép ảnh, viền box phải bám sát mép ảnh (toạ độ 0 hoặc kích thước tối đa của ảnh).

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): `instance_id: "kitchen-001"`, `class_name: "person"`, `score: 0.899318`, `polygon_point_count: 348`, `polygon_xy: [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0], [439.0, 73.0], [438.0, 74.0], ...]`.
- Polygon bổ sung chi tiết gì so với box? Polygon cung cấp đường bao hình học chính xác đến từng pixel theo hình dạng thực tế của đối tượng, loại bỏ hoàn toàn các vùng nền và không gian trống bên ngoài vật thể mà bounding box hình chữ nhật bắt buộc phải bao quanh.
- `instance_id` dùng để làm gì và không phải loại ID nào? `instance_id` dùng để định danh và phân biệt riêng biệt từng cá thể đối tượng trong cùng một bức ảnh (kể cả khi chúng cùng thuộc một class); đây là ID theo ngữ cảnh bài lab (`<sample_id>-NNN`), không phải `class_id` trong danh mục cũng không phải object tracking ID xuyên suốt các frame video.
- Đề xuất một quy tắc biên mask: Đường biên mask phải chạy khít theo viền vật thể với sai số không quá 1-2 pixel, bám đúng các đường cong, chi tiết lồi/lõm và tách biệt hoàn toàn phần rỗng bên trong (nếu có).
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Guideline/escalation cần quy định: (1) Cách xử lý ranh giới khi hai vật thể cùng màu sắc/chất liệu tiếp xúc nhau; (2) Có phân đoạn phần chi tiết bị che khuất thành nhiều đa giác con (multi-polygon) thuộc cùng 1 instance hay không; (3) Tiêu chuẩn xử lý vùng bóng đổ và viền mờ do chuyển động (motion blur).

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | `class_id`, `class_name` (cấp ảnh) | Ảnh chứa nhiều chủ thể khác nhau (ví dụ xe bus, ô tô, người đi bộ trong ảnh `traffic`), khó chọn 1 nhãn duy nhất. | Đối chiếu guideline xác định chủ thể trung tâm/chiếm diện tích chính để gán nhãn; nếu không rõ thì escalate. | Kiểm tra nhãn được gán có khớp đúng chủ thể trọng tâm theo guideline và đúng taxonomy không. |
| Phát hiện vật thể | Bounding box `[x_min, y_min, x_max, y_max]` + `class_id` | Vật thể bị che khuất một phần (người đứng sau quầy bếp) hoặc xếp chồng lấn (chén bát trên bàn). | Vẽ box ôm sát viền nhìn thấy của từng thực thể độc lập, không bỏ sót đối tượng nhỏ hoặc cắt mép. | Soi độ ôm sát của box (không quá rộng/hẹp), kiểm tra không trùng lặp (duplicate) và không bỏ sót. |
| Instance segmentation | Instance ID + `polygon_xy` theo pixel + `class_id` | Ranh giới mờ giữa vật thể và phông nền; phần cơ thể/vật thể bị vật cản cắt rời thành nhiều mảng. | Dùng công cụ polygon chấm điểm khít theo biên dạng thực tế; tách instance riêng cho từng cá thể cùng lớp. | Zoom cận cảnh kiểm tra độ mượt/độ bám biên của mask, kiểm tra không bị lấn nền hoặc mất chi tiết. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Tuyệt đối không tải lên dữ liệu chứa thông tin định danh cá nhân (PII), hình ảnh nhạy cảm hoặc bí mật dự án lên các nền tảng công cộng trái quy định; tuân thủ đúng quyền truy cập và lưu trữ bảo mật.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Giảng viên / Mentor phụ trách bài thực hành hoặc Quality Control Lead / Project Manager.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
