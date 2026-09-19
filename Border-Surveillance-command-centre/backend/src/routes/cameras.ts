import { Router } from 'express';
import { requireAuth } from '../middleware/auth';
import { cameraController } from '../controllers/cameraController';

const router = Router();

// Only authenticated users can request a stream token
router.use(requireAuth);

router.get('/token', cameraController.getStreamToken);

export default router;
