import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetection=0.5, minTrackCon=0.5):
        """
        Initializes the FaceMeshDetector.

        Parameters:
        staticMode (bool): Whether to treat input images as a batch of static images.
        maxFaces (int): Maximum number of faces to detect.
        minDetection (float): Minimum detection confidence threshold.
        minTrackCon (float): Minimum tracking confidence threshold.
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetection = minDetection
        self.minTrackCon = minTrackCon

        # Mediapipe setup
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

        # Initialize FaceMesh model
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetection,
            min_tracking_confidence=self.minTrackCon
        )

    def findFaceMesh(self, img, draw=True):
        """
        Detects and draws face mesh landmarks on the given image.

        Parameters:
        img (ndarray): Input image.
        draw (bool): Whether to draw the landmarks on the image.

        Returns:
        img: Image with or without face mesh landmarks.
        faces: List of detected face landmarks with coordinates.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        faces = []  # Stores all face landmarks

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        faceLms,
                        self.mpFaceMesh.FACEMESH_TESSELATION,
                        self.drawSpec,
                        self.drawSpec
                    )

                # Extract face landmark coordinates
                face = []
                for id,lm in faceLms.landmark:
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,255, 0), 1)
                    face.append([x, y])
                faces.append(face)

        return img, faces


def main():
    """Dummy testing code. Runs only if module is executed directly."""
    # Change to your video file if needed
    # cap = cv2.VideoCapture("Videos/2.mp4")
    cap = cv2.VideoCapture(0)  # Use webcam

    pTime = 0
    detector = FaceMeshDetector(maxFaces=4)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame. Exiting...")
            break

        img, faces = detector.findFaceMesh(img, draw=True)

        # If faces detected, print basic info
        if faces:
            print(f"Number of faces: {len(faces)} | First face landmarks: {len(faces[0])}")

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        img = cv2.resize(img, (1000, 600))
        cv2.imshow("Face Mesh", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ✅ This ensures the main() runs only when this module is executed directly
if __name__ == "__main__":
    main()
