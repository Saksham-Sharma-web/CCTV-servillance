import { Router } from 'express';
import { requireAuth, requireAdmin } from '../middleware/auth';
import { userController } from '../controllers/userController';

const router = Router();

// All routes require authenticated admin
router.use(requireAuth, requireAdmin);

router.get('/', userController.getUsers);
router.post('/', userController.createUser);
router.patch('/:id/role', userController.updateRole);
router.patch('/:id', userController.updateUser);
router.post('/:id/disable', userController.disableUser);
router.post('/:id/enable', userController.enableUser);
router.delete('/:id', userController.deleteUser);
router.post('/:id/password-reset', userController.resetPassword);

export default router;
