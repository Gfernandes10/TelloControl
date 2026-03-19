# Como Calibrar a Câmera do Tello

## 🎯 Por que calibrar?

Os valores atuais no código são **aproximados**. Para usar com:
- ✅ AprilTag detection
- ✅ ArUco markers
- ✅ Visual SLAM
- ✅ Pose estimation precisa

Você precisa de **calibração real**.

## 📋 O que você precisa

1. **Tabuleiro de xadrez** impresso (recomendado: 8x6 quadrados, 108mm cada)
   - [Baixar PDF](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration?action=AttachFile&do=get&target=check-108.pdf)
   - Imprimir em papel A4 e colar em superfície rígida

2. **Pacote de calibração do ROS 2:**
```bash
sudo apt install ros-humble-camera-calibration ros-humble-camera-calibration-parsers
```

## 🔧 Passo a Passo

### 1. Iniciar o nó do Tello

```bash
cd ~/TelloControl
source install/setup.bash
ros2 launch tello_interface launch_real_tello.py
```

### 2. Verificar os tópicos

Em outro terminal:
```bash
ros2 topic list | grep tello
# Deve aparecer:
# /tello/image_raw
# /tello/camera_info
```

### 3. Executar ferramenta de calibração

```bash
ros2 run camera_calibration cameracalibrator \
    --size 8x6 \
    --square 0.108 \
    --no-service-check \
    image:=/tello/image_raw \
    camera:=/tello
```

**Parâmetros:**
- `--size 8x6`: Número de **cantos internos** (não quadrados)
- `--square 0.108`: Tamanho de cada quadrado em metros (108mm)
- `image:=/tello/image_raw`: Tópico da imagem

### 4. Processo de calibração

Uma janela vai abrir mostrando:
- **X**: Cobertura horizontal (mover tabuleiro esquerda/direita)
- **Y**: Cobertura vertical (mover tabuleiro cima/baixo)
- **Size**: Diferentes distâncias (perto/longe)
- **Skew**: Diferentes ângulos/rotações

**Como fazer:**
1. Segure o tabuleiro impresso na frente da câmera do Tello
2. Mova-o lentamente em diferentes posições:
   - Esquerda, direita, cima, baixo
   - Perto e longe
   - Inclinado em vários ângulos
3. As barras X, Y, Size, Skew vão preenchendo
4. Quando todas estiverem **verdes**, o botão **CALIBRATE** fica disponível
5. Clique em **CALIBRATE** e aguarde (pode demorar alguns minutos)
6. Clique em **SAVE** para salvar os resultados

### 5. Usar os resultados

A calibração será salva em `/tmp/calibrationdata.tar.gz`

Extrair e visualizar:
```bash
cd /tmp
tar -xzf calibrationdata.tar.gz
cat ost.yaml
```

Copiar para seu pacote:
```bash
cp ost.yaml ~/TelloControl/src/dji_tello_driver/config/tello_camera_calibrated.yaml
```

### 6. Atualizar o código

Edite `tello_node.py` para carregar do arquivo YAML:

```python
from camera_calibration_parsers import readCalibration

def setup_camera_info(self):
    # Carregar calibração do arquivo
    yaml_path = '/caminho/para/tello_camera_calibrated.yaml'
    camera_name, camera_info = readCalibration(yaml_path)
    self.camera_info = camera_info
    self.camera_info.header.frame_id = 'tello_camera'
```

## 📊 Valores esperados (referência)

Após calibração, valores típicos do Tello:
- **fx, fy**: 900-950 pixels
- **cx**: ~480 pixels (centro horizontal)
- **cy**: ~360 pixels (centro vertical)
- **k1, k2, k3**: Coeficientes de distorção radial
- **p1, p2**: Coeficientes de distorção tangencial

## ✅ Verificar qualidade

Após calibração, verifique:
```bash
# Erro de reprojeção deve ser < 0.5 pixels (idealmente < 0.3)
# Aparece no terminal durante a calibração
```

## 🎥 Alternativa: Usar OpenCV

Se preferir usar Python puro:
```python
import cv2
import numpy as np
import glob

# Preparar pontos do tabuleiro
objp = np.zeros((6*8, 3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 0.108

objpoints = []  # Pontos 3D no mundo real
imgpoints = []  # Pontos 2D na imagem

# Capturar várias imagens do tabuleiro
# ... processar com cv2.findChessboardCorners()

# Calibrar
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (960, 720), None, None)

# mtx = matriz K (camera_matrix)
# dist = coeficientes de distorção
```

## 🔗 Recursos

- [ROS 2 Camera Calibration Tutorial](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration)
- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Calibration Target Generator](https://calib.io/pages/camera-calibration-pattern-generator)

## ⚠️ Dicas

1. Use **iluminação uniforme** (evitar reflexos)
2. Tabuleiro deve estar **completamente plano** (não ondulado)
3. Capture **pelo menos 30 poses diferentes**
4. Movimente **devagar** para evitar motion blur
5. Cubra **toda a área da imagem** uniformemente
