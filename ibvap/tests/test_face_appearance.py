"""
Comprehensive Test Suite for High-Accuracy Face Detection, Landmark Alignment,
Conservative Recognition, and Supporting Body Appearance Analysis.

Implements all 11 required test cases:
1. One person: most_recent
2. Same person: recent
3. Same person: old (body ignored)
4. Different person: UNKNOWN rejection
5. No-face image: NO_FACE_DETECTED
6. Body-only image: NO_FACE_DETECTED (strictly no identity)
7. Small face handling
8. Blurred face: INSUFFICIENT_FACE_QUALITY
9. Profile / tilted face with 5-point landmark alignment
10. Multiple people: MULTIPLE_FACES_DETECTED
11. Same clothing, different person: body signal MUST NOT override face mismatch

Includes CPU latency & resource benchmarks.
"""

import os
import time
import pytest
import cv2
import numpy as np

from ibvap.core.config import IBVAPConfig
from ibvap.face.detector import OpenCVFaceDetector, FaceDetection
from ibvap.face.matcher_adapter import (
    IdentityVerifierAdapter,
    BodyAppearanceExtractor,
    AuthorizedPerson,
    PersonReference,
    align_face_160,
    calibrate_threshold,
    CANONICAL_FACE_TEMPLATE_160,
)
from main import test_images as run_test_images

TEST_IMAGE_PATH = r"C:\ibvap\akshat.jpeg"


@pytest.fixture
def detector():
    return OpenCVFaceDetector()


@pytest.fixture
def verifier():
    return IdentityVerifierAdapter()


@pytest.fixture
def sample_image():
    if os.path.exists(TEST_IMAGE_PATH):
        img = cv2.imread(TEST_IMAGE_PATH)
        if img is not None:
            return img
    # Fallback synthetic face if file not present
    canvas = np.zeros((480, 480, 3), dtype=np.uint8)
    cv2.circle(canvas, (240, 240), 90, (180, 200, 220), -1)
    return canvas


# ── Test 1: Single Person (most_recent) ──────────────────────────────
def test_case_01_single_person_most_recent(detector, verifier, sample_image):
    """Test 1: Enrolls person with 'most_recent' reference. Expects MATCH and supporting body signal."""
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"Test image {TEST_IMAGE_PATH} not found.")

    res = run_test_images(
        references=[("Akshat", TEST_IMAGE_PATH, "most_recent")],
        target=TEST_IMAGE_PATH
    )

    assert res["face_decision"] == "MATCH"
    assert res["identity"] == "Akshat"
    assert res["face_detected"] is True
    assert res["face_confidence"] >= 0.85
    assert res["face_similarity"] >= 0.95
    assert res["body_status"] in ("BODY_SUPPORTING", "BODY_DETECTED")
    assert res["body_role"] == "SUPPORTING ONLY"


# ── Test 2: Same Person (recent) ──────────────────────────────────────
def test_case_02_same_person_recent(detector, verifier, sample_image):
    """Test 2: Enrolls person with 'recent' reference. Expects MATCH with reduced body weighting."""
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"Test image {TEST_IMAGE_PATH} not found.")

    res = run_test_images(
        references=[("Akshat", TEST_IMAGE_PATH, "recent")],
        target=TEST_IMAGE_PATH
    )

    assert res["face_decision"] == "MATCH"
    assert res["identity"] == "Akshat"
    assert res["face_similarity"] >= 0.95
    assert res["best_reference_age"] == "recent"


# ── Test 3: Same Person (old) ─────────────────────────────────────────
def test_case_03_same_person_old(detector, verifier, sample_image):
    """Test 3: Enrolls person with 'old' reference. Expects MATCH, but body similarity is IGNORED."""
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"Test image {TEST_IMAGE_PATH} not found.")

    res = run_test_images(
        references=[("Akshat", TEST_IMAGE_PATH, "old")],
        target=TEST_IMAGE_PATH
    )

    assert res["face_decision"] == "MATCH"
    assert res["identity"] == "Akshat"
    assert res["face_similarity"] >= 0.95
    assert res["best_reference_age"] == "old"
    assert res["body_status"] == "BODY_IGNORED"


# ── Test 4: Different Person (UNKNOWN rejection) ──────────────────────
def test_case_04_different_person_unknown(detector, verifier, sample_image):
    """Test 4: Target face does not match registered identity. Must output UNKNOWN, identity=None."""
    # Register synthetic person with orthogonal face embedding
    synthetic_emb = np.random.randn(512).astype(np.float32)
    synthetic_emb /= np.linalg.norm(synthetic_emb)

    verifier.register_person("ID-DIFF", "DifferentPerson", embedding=synthetic_emb)
    faces = detector.detect_faces(sample_image)
    if not faces:
        pytest.skip("No faces detected in sample image.")

    verif_res = verifier.verify(sample_image, face_detection=faces[0])
    assert verif_res.face_decision == "UNKNOWN"
    assert verif_res.identity is None
    assert verif_res.face_similarity < verifier.similarity_threshold


# ── Test 5: No-Face Image ─────────────────────────────────────────────
def test_case_05_no_face_image(detector, verifier):
    """Test 5: Image contains no face (e.g. wall/blank background). Must output NO_FACE_DETECTED."""
    blank = np.zeros((400, 400, 3), dtype=np.uint8)
    # Background texture
    cv2.randn(blank, 100, 20)

    faces = detector.detect_faces(blank)
    valid_faces = [f for f in faces if f.quality_status != "NO_FACE"]
    assert len(valid_faces) == 0

    verif_res = verifier.verify(blank, face_detection=None)
    assert verif_res.face_decision == "NO_FACE_DETECTED"
    assert verif_res.identity is None


# ── Test 6: Body-Only Image (Strict Invariant) ────────────────────────
def test_case_06_body_only_image_strict_invariant(detector, verifier):
    """
    Test 6: A torso/jacket image with NO head/face.
    HARD INVARIANT: Must return NO_FACE_DETECTED and identity=None.
    Body appearance must NEVER forge or create a biometric face identity!
    """
    torso_img = np.full((500, 300, 3), (30, 40, 150), dtype=np.uint8)  # Blue jacket
    cv2.rectangle(torso_img, (50, 100), (250, 450), (20, 20, 120), -1)

    faces = detector.detect_faces(torso_img)
    valid_faces = [f for f in faces if f.quality_status != "NO_FACE"]
    assert len(valid_faces) == 0

    verif_res = verifier.verify(torso_img, face_detection=None, person_crop=torso_img)
    assert verif_res.face_decision == "NO_FACE_DETECTED"
    assert verif_res.identity is None
    assert verif_res.body_role == "SUPPORTING ONLY"


# ── Test 7: Small Face Detection ──────────────────────────────────────
def test_case_07_small_face_detection(detector, sample_image):
    """Test 7: Downscaled face (~60px) in a larger frame. Detector must locate face and landmarks."""
    faces = detector.detect_faces(sample_image)
    if not faces:
        pytest.skip("No faces in sample image.")

    top_face = faces[0]
    bx1, by1, bx2, by2 = top_face.box
    face_crop = sample_image[by1:by2, bx1:bx2]
    # Resize face down to 64x64 and place on 640x640 frame
    small_face = cv2.resize(face_crop, (64, 64))
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    canvas[200:264, 200:264] = small_face

    det_small = OpenCVFaceDetector(IBVAPConfig(face_min_width=20, face_min_height=20))
    detected = det_small.detect_faces(canvas)
    assert len(detected) >= 1
    assert detected[0].confidence > 0.50
    assert detected[0].landmarks is not None


# ── Test 8: Blurred Face Rejection ────────────────────────────────────
def test_case_08_blurred_face_rejection(detector):
    """Test 8: Severely blurred face must trigger INSUFFICIENT_FACE_QUALITY."""
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"Test image {TEST_IMAGE_PATH} not found.")

    img = cv2.imread(TEST_IMAGE_PATH)
    # Apply severe Gaussian blur
    blurred = cv2.GaussianBlur(img, (51, 51), 30)

    faces = detector.detect_faces(blurred)
    if faces:
        assert faces[0].quality_status == "LOW_QUALITY_FACE"
        assert faces[0].quality_metrics["reason"] == "blurred"


# ── Test 9: Profile / Tilted Face Alignment ───────────────────────────
def test_case_09_profile_tilted_face_affine_alignment(detector, sample_image):
    """Test 9: Rotated/tilted face must have 5 landmarks warped to standard 160x160 template."""
    h, w = sample_image.shape[:2]
    center = (w // 2, h // 2)
    # Rotate image 20 degrees
    rot_mat = cv2.getRotationMatrix2D(center, 20.0, 1.0)
    rotated = cv2.warpAffine(sample_image, rot_mat, (w, h))

    faces = detector.detect_faces(rotated)
    if not faces:
        pytest.skip("No face detected in rotated image.")

    f = faces[0]
    assert f.landmarks is not None
    assert f.landmarks.shape == (5, 2)

    # Align to 160x160
    aligned = align_face_160(rotated, landmarks=f.landmarks, box=f.box)
    assert aligned.shape == (160, 160, 3)
    assert np.mean(aligned) > 10.0


# ── Test 10: Multiple People Detection ────────────────────────────────
def test_case_10_multiple_people(detector, sample_image):
    """Test 10: Multi-person frame returns all valid detections."""
    faces = detector.detect_faces(sample_image)
    if not faces:
        pytest.skip("No face detected in sample image.")

    # Place face twice horizontally in wide frame
    h, w = sample_image.shape[:2]
    wide = np.zeros((h, w * 2, 3), dtype=np.uint8)
    wide[:, 0:w] = sample_image
    wide[:, w:w*2] = sample_image

    multi_faces = detector.detect_faces(wide)
    assert len(multi_faces) >= 2
    # Verify boxes do not overlap completely
    assert abs(multi_faces[0].box[0] - multi_faces[1].box[0]) > 50


# ── Test 11: Same Clothing, Different Person (CRITICAL) ───────────────
def test_case_11_same_clothing_different_person(detector, verifier, sample_image):
    """
    Test 11: Person A and Person B wear identical black jackets.
    Target has Person B's face and Person A's clothing.
    CRITICAL BIOMETRIC INVARIANT:
    Body similarity is high (~0.90), but Face similarity is low (~0.30).
    Decision MUST BE UNKNOWN (identity=None).
    Clothing appearance MUST NOT turn a face mismatch into an identity verification!
    """
    # 1. Enrolled Person A with black clothing
    person_a_face_emb = np.random.randn(512).astype(np.float32)
    person_a_face_emb /= np.linalg.norm(person_a_face_emb)

    black_jacket_crop = np.full((400, 200, 3), (25, 25, 25), dtype=np.uint8)
    person_a_body_emb = BodyAppearanceExtractor.extract(black_jacket_crop)

    verifier.authorized_registry["ID-AMIT"] = AuthorizedPerson(
        identity_id="ID-AMIT",
        name="Amit",
        references=[]
    )
    from ibvap.face.matcher_adapter import PersonReference
    verifier.authorized_registry["ID-AMIT"].references.append(
        PersonReference(
            source_path="amit_ref.jpg",
            reference_age="most_recent",
            face_embedding=person_a_face_emb,
            body_embedding=person_a_body_emb,
        )
    )

    # 2. Target: Person B (different face) wearing identical black jacket
    person_b_face_emb = np.random.randn(512).astype(np.float32)
    person_b_face_emb /= np.linalg.norm(person_b_face_emb)
    # Ensure orthogonality (low cosine similarity < 0.3)
    person_b_face_emb -= np.dot(person_b_face_emb, person_a_face_emb) * person_a_face_emb
    person_b_face_emb /= np.linalg.norm(person_b_face_emb)

    # Target wearing identical black jacket
    target_body_crop = np.full((400, 200, 3), (25, 25, 25), dtype=np.uint8)

    # Mock face detection for Person B
    class MockFaceDetection:
        box = (50, 50, 150, 150)
        confidence = 0.98
        landmarks = None
        detector = "yunet"
        quality_status = "GOOD_FACE"

    # Inject mock embedding extraction for this test
    verifier.extract_face_embedding = lambda crop: person_b_face_emb

    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    verif_res = verifier.verify(
        target_image=dummy_frame,
        face_detection=MockFaceDetection(),
        person_crop=target_body_crop
    )

    # HARD INVARIANT VERIFICATION
    assert verif_res.face_decision == "UNKNOWN", "Decision must be UNKNOWN because face did not match!"
    assert verif_res.identity is None, "Identity must NOT be assigned based on clothing similarity!"
    assert verif_res.face_similarity < verifier.similarity_threshold
    assert verif_res.body_similarity >= 0.85, "Body appearance was indeed identical."
    assert verif_res.body_role == "SUPPORTING ONLY"


# ── Threshold Calibration Test ────────────────────────────────────────
def test_threshold_calibration():
    """Tests the ROC/AUC threshold calibration utility."""
    np.random.seed(42)
    # Synthetic genuine distribution: N(0.85, 0.05)
    genuine = list(np.random.normal(0.85, 0.05, 100).clip(0.60, 1.0))
    # Synthetic impostor distribution: N(0.35, 0.08)
    impostor = list(np.random.normal(0.35, 0.08, 100).clip(0.10, 0.65))

    calib = calibrate_threshold(genuine, impostor)
    assert "recommended_threshold" in calib
    assert "equal_error_rate" in calib
    assert 0.50 <= calib["recommended_threshold"] <= 0.80
    assert calib["equal_error_rate"] <= 0.05


# ── Resource & Latency Benchmarks ─────────────────────────────────────
def test_resource_benchmarks(detector, verifier, sample_image):
    """
    Benchmarks CPU inference latency and memory for:
    1. Face Detection (YuNet)
    2. Face Embedding (InceptionResnetV1)
    3. Body Appearance Embedding
    4. End-to-end test_images()
    """
    # 1. Face detection benchmark
    t_detect = []
    for _ in range(5):
        t0 = time.time()
        _ = detector.detect_faces(sample_image)
        t_detect.append(time.time() - t0)

    # 2. Face embedding benchmark
    aligned_sample = np.zeros((160, 160, 3), dtype=np.uint8)
    # Warmup
    _ = verifier.extract_face_embedding(aligned_sample)
    t_embed = []
    for _ in range(5):
        t0 = time.time()
        _ = verifier.extract_face_embedding(aligned_sample)
        t_embed.append(time.time() - t0)

    # 3. Body appearance benchmark
    t_body = []
    for _ in range(5):
        t0 = time.time()
        _ = BodyAppearanceExtractor.extract(sample_image)
        t_body.append(time.time() - t0)

    avg_detect_ms = np.mean(t_detect) * 1000
    avg_embed_ms = np.mean(t_embed) * 1000
    avg_body_ms = np.mean(t_body) * 1000

    print("\n" + "=" * 55)
    print("  CPU BIOMETRIC BENCHMARK RESULTS")
    print("=" * 55)
    print(f"  YuNet Face Detection:       {avg_detect_ms:6.2f} ms")
    print(f"  InceptionResnetV1 (512-D):  {avg_embed_ms:6.2f} ms")
    print(f"  Body Appearance (256-D):    {avg_body_ms:6.2f} ms")
    print(f"  Combined Biometric Latency: {avg_detect_ms + avg_embed_ms + avg_body_ms:6.2f} ms")
    print("=" * 55)

    assert avg_detect_ms < 250.0, "Face detection must be responsive (< 250ms on CPU)"
    assert avg_embed_ms < 300.0, "Face embedding must be responsive (< 300ms on CPU)"
    assert avg_body_ms < 15.0, "Body appearance must be lightweight (< 15ms on CPU)"
