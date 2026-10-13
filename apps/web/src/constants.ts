export const LABELS = ['CLEAN', 'OFFENSIVE', 'HATE'] as const
export type Label = (typeof LABELS)[number]

export const LABEL_VI: Record<Label, string> = {
  CLEAN: 'Bình thường',
  OFFENSIVE: 'Xúc phạm',
  HATE: 'Thù ghét',
}

export const LABEL_COLOR: Record<Label, string> = {
  CLEAN: '#16a34a',
  OFFENSIVE: '#f59e0b',
  HATE: '#dc2626',
}

export const BRAND_RGB = '21,101,192'

export const TEAM = {
  name: 'Nhóm 5',
  supervisor: 'PGS.TS. Hoàng Văn Dũng',
  members: [
    { id: '2611318', name: 'Vòng Vĩnh Shìn' },
    { id: '2611302', name: 'Trần Nhật Duật' },
    { id: '2611320', name: 'Hồ Đức Thắng' },
  ],
}

export const EXAMPLES = [
  'Video này hay quá, cảm ơn bạn đã chia sẻ nhé',
  'Thằng ngu này nói chuyện chán vãi',
  'Bọn mày là lũ súc vật không hơn không kém',
]

// Định dạng số theo chuẩn Việt Nam: dấu phẩy thập phân, dấu chấm phân cách hàng nghìn.
export const dec = (v: number, digits: number) =>
  v.toLocaleString('vi-VN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
export const pct = (v: number, digits = 1) => `${dec(v * 100, digits)}%`
export const num = (v: number) => v.toLocaleString('vi-VN')
/** Độ rộng CSS phải dùng dấu chấm thập phân, nên không dùng pct(). */
export const cssWidth = (v: number) => `${v * 100}%`

export const EMPTY_MSG = 'Câu trống sau khi làm sạch (chỉ có URL, mention hoặc ký tự đặc biệt).'
