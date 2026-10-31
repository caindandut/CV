export type MediaType = "image" | "video";
export type Status = "queued" | "processing" | "completed" | "failed";
export interface Detection {
  class_id: number;
  class_name: string;
  label: string;
  confidence: number;
  xyxy: number[];
}
export interface Job {
  id: string;
  original_filename: string;
  media_type: MediaType;
  status: Status;
  progress: number;
  model_identifier: string;
  imgsz: number;
  confidence_threshold: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  detections: Detection[] | null;
}
export interface JobPage {
  items: Job[];
  page: number;
  page_size: number;
  total: number;
}
export interface Health {
  database_ready: boolean;
  model_ready: boolean;
  model_message: string;
  device: string | null;
  font_ready: boolean;
  ffmpeg_ready: boolean;
  imgsz: number;
  confidence_threshold: number;
  max_upload_mb: number;
  startup_complete: boolean;
}
export const statusText: Record<Status, string> = {
  queued: "Đang chờ",
  processing: "Đang xử lý",
  completed: "Hoàn thành",
  failed: "Thất bại",
};
export const confidence = (value: number) =>
  `${(value * 100).toLocaleString("vi-VN", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`;
export async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.error?.message ?? "Không thể thực hiện yêu cầu. Vui lòng thử lại.",
    );
  }
  return response.json();
}
export const fileUrl = (id: string, output: boolean, download = false) =>
  `/api/jobs/${id}/${output ? "output" : "input"}${download ? "?download=true" : ""}`;
