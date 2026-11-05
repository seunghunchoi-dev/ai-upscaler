# AI 영상·사진 업스케일러 (베타)

저화질 영상과 사진의 **압축 깨짐 복원 → AI 업스케일 → 얼굴 복원 → 프레임 보간(60fps)** 을 한 번에 처리하는 Windows 프로그램입니다.

> **베타 버전입니다.** 오류가 있을 수 있으니 원본 파일은 꼭 따로 보관하세요.

## 주요 기능

- 영상 자동 분석 후 추천 설정 제공
- Real-ESRGAN 기반 압축 깨짐 제거·업스케일 (최대 8K 출력)
- GFPGAN 기반 얼굴 복원 (선택 사항)
- RIFE 기반 프레임 보간 (60fps·120fps 등)
- AI 없이 FFmpeg 필터만 쓰는 빠른 처리
- 사진 일괄 처리, 원본/결과 비교 미리보기
- 중단해도 이어서 처리
- Intel GPU·NPU, NVIDIA, AMD GPU 지원

## 사용 방법

1. [Releases](https://github.com/seunghunchoi-dev/ai-upscaler/releases) 에서 최신 베타 설치 파일을 받아 실행합니다.
2. 처음 실행할 때 나오는 **추가 구성요소** 창에서 FFmpeg과 얼굴 복원 모델을 받습니다.

두 구성요소는 라이선스 때문에 설치 파일에 넣지 않았으며, 사용자가 공식 제공처에서 직접 받습니다.

**[자세한 설치 방법과 사용 설명서는 Wiki에 있습니다.](https://github.com/seunghunchoi-dev/ai-upscaler/wiki)**

## 라이선스

**소스 코드는 공개되어 있지만 오픈소스가 아닙니다.**

| 가능 | 불가능 (저작권자의 사전 서면 허락 필요) |
|---|---|
| 내려받아 개인적·비상업적으로 사용 | 소스 또는 프로그램 수정, 2차 저작물 작성 |
| 소스 열람, 수정 없이 본인 PC에서 빌드·실행 | 다른 곳에 다시 올리거나 배포 |
| 이 저장소 링크 공유 | 판매 등 상업적 이용 |

전문은 [LICENSE.txt](LICENSE.txt), 제3자 구성 요소는 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) 를 참고하세요.

처리할 권리가 있는 영상과 사진에만 사용하세요. 콘텐츠 사용에 따른 책임은 사용자에게 있습니다.

Copyright (c) 2026 최승훈. All rights reserved.
