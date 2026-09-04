import { Request, Response } from 'express';
import jwt from 'jsonwebtoken';

const STREAM_SECRET = process.env.STREAM_SECRET || 'v8!x@9Pq2L#mZ5$k*RyT^7&wF4(cD1%h';

export const cameraController = {
  async getStreamToken(req: Request, res: Response) {
    try {
      // The user is already authenticated via the requireAuth middleware.
      // We can generate a short-lived token specifically for streaming.
      const userId = req.user.id;
      
      // Token expires in 5 minutes (300 seconds)
      const token = jwt.sign({ sub: userId, type: 'video_stream' }, STREAM_SECRET, { expiresIn: '5m' });

      return res.json({ success: true, data: { token } });
    } catch (error: any) {
      console.error('Error generating stream token:', error);
      return res.status(500).json({ success: false, error: { message: 'Failed to generate token' } });
    }
  }
};
