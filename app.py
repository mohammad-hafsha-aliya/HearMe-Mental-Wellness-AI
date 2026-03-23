"""
HearMe Mental Wellness AI Companion
A comprehensive Flask application with database, authentication, and advanced features
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
from textblob import TextBlob
import numpy as np
from collections import Counter
import json
import os
import re

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'hearme-secret-key-change-in-production-2026')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///hearme.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set True in production with HTTPS

# Initialize extensions
CORS(app, supports_credentials=True)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login_page'

# ==================== DATABASE MODELS ====================

class User(UserMixin, db.Model):
    """User account model"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    theme = db.Column(db.String(20), default='light')

    # Relationships
    journal_entries = db.relationship('JournalEntry', backref='user', lazy=True, cascade='all, delete-orphan')
    mood_logs = db.relationship('MoodLog', backref='user', lazy=True, cascade='all, delete-orphan')
    goals = db.relationship('Goal', backref='user', lazy=True, cascade='all, delete-orphan')
    chat_messages = db.relationship('ChatMessage', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'theme': self.theme,
            'created_at': self.created_at.isoformat()
        }


class JournalEntry(db.Model):
    """Journal entry with sentiment analysis"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    polarity = db.Column(db.Float)
    subjectivity = db.Column(db.Float)
    mood = db.Column(db.String(20))
    emotions = db.Column(db.Text)

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'created_at': self.created_at.isoformat(),
            'sentiment': {
                'polarity': self.polarity,
                'subjectivity': self.subjectivity,
                'mood': self.mood,
                'color': get_mood_color(self.mood)
            },
            'emotions': json.loads(self.emotions) if self.emotions else {}
        }


class MoodLog(db.Model):
    """Quick mood check-in"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    mood = db.Column(db.String(20), nullable=False)
    energy_level = db.Column(db.Integer)
    stress_level = db.Column(db.Integer)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'mood': self.mood,
            'energy_level': self.energy_level,
            'stress_level': self.stress_level,
            'notes': self.notes,
            'created_at': self.created_at.isoformat()
        }


class Goal(db.Model):
    """Wellness goals"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(50))
    target_value = db.Column(db.Integer)
    current_value = db.Column(db.Integer, default=0)
    unit = db.Column(db.String(20))
    deadline = db.Column(db.Date)
    completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        progress = (self.current_value / self.target_value * 100) if self.target_value and self.target_value > 0 else 0
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'target_value': self.target_value,
            'current_value': self.current_value,
            'unit': self.unit,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'completed': self.completed,
            'progress': min(progress, 100),
            'created_at': self.created_at.isoformat()
        }


class ChatMessage(db.Model):
    """AI chatbot conversation"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    sentiment_mood = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_message': self.user_message,
            'bot_response': self.bot_response,
            'sentiment_mood': self.sentiment_mood,
            'created_at': self.created_at.isoformat()
        }


# ==================== LOGIN MANAGER ====================

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))  # Fixed: use db.session.get (query.get deprecated)

@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({'error': 'Unauthorized'}), 401


# ==================== HELPER FUNCTIONS ====================

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, 'Password must be at least 8 characters'
    if not re.search(r'[A-Za-z]', password):
        return False, 'Password must contain at least one letter'
    return True, None

def validate_username(username):
    """Validate username format"""
    if len(username) < 3:
        return False, 'Username must be at least 3 characters'
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, 'Username can only contain letters, numbers, and underscores'
    return True, None

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.5:
        mood = 'very_positive'
    elif polarity > 0:
        mood = 'positive'
    elif polarity < -0.5:
        mood = 'very_negative'
    elif polarity < 0:
        mood = 'negative'
    else:
        mood = 'neutral'

    return {
        'polarity': round(polarity, 3),
        'subjectivity': round(subjectivity, 3),
        'mood': mood
    }

def extract_emotions(text):
    """Extract emotional keywords"""
    positive_emotions = ['happy', 'joy', 'love', 'excited', 'grateful', 'peaceful',
                         'content', 'hopeful', 'proud', 'inspired', 'confident', 'relaxed']
    negative_emotions = ['sad', 'angry', 'anxious', 'depressed', 'worried', 'stressed',
                         'lonely', 'frustrated', 'hurt', 'scared', 'guilty', 'ashamed']

    text_lower = text.lower()
    found_positive = [em for em in positive_emotions if em in text_lower]
    found_negative = [em for em in negative_emotions if em in text_lower]

    return {
        'positive': found_positive,
        'negative': found_negative
    }

def get_mood_color(mood):
    """Get color for mood"""
    colors = {
        'very_positive': '#10b981',
        'positive': '#84cc16',
        'neutral': '#f59e0b',
        'negative': '#ef4444',
        'very_negative': '#dc2626'
    }
    return colors.get(mood, '#6b7280')

def generate_chatbot_response(user_message, sentiment_data):
    """Generate intelligent chatbot responses"""
    message_lower = user_message.lower()
    mood = sentiment_data['mood']

    # Greetings
    if any(g in message_lower for g in ['hello', 'hi', 'hey']):
        return "Hello! I'm HearMe, your wellness companion. How are you feeling today? 💚"

    # Help
    if any(w in message_lower for w in ['help', 'what can you do']):
        return ("I'm here to support you! I can:\n\n"
                "✓ Listen and provide empathetic responses\n"
                "✓ Analyze your emotions\n"
                "✓ Suggest coping strategies\n"
                "✓ Recommend music and meditation\n"
                "✓ Help track your mood\n\n"
                "Feel free to share what's on your mind! 😊")

    # Anxiety
    if any(k in message_lower for k in ['anxious', 'anxiety', 'stressed', 'worried']):
        return ("I hear you're feeling anxious. Try this:\n\n"
                "🌬️ 4-7-8 breathing: Inhale 4, hold 7, exhale 8\n"
                "🧘 5-4-3-2-1 grounding technique\n"
                "💚 Remember: You're stronger than you think\n\n"
                "Would you like calming music?")

    # Depression
    if any(w in message_lower for w in ['depressed', 'sad', 'hopeless', 'lonely']):
        return ("I'm sorry you're going through this. Remember:\n\n"
                "💙 You are not alone\n"
                "🌱 Small steps matter\n"
                "🤝 Consider reaching out to a professional\n\n"
                "**Crisis Resources:**\n"
                "• 988 Suicide & Crisis Lifeline\n"
                "• Crisis Text Line: HOME to 741741\n\n"
                "I'm here to listen. What would help right now?")

    # Sleep
    if any(w in message_lower for w in ['sleep', 'insomnia', 'tired']):
        return ("Sleep is crucial for wellness:\n\n"
                "🌙 Consistent bedtime routine\n"
                "📵 No screens 1 hour before bed\n"
                "🧘 Progressive relaxation\n"
                "☕ Limit caffeine after 2 PM\n\n"
                "I can suggest relaxing music!")

    # Mood-based responses
    if mood == 'very_negative':
        return ("I sense you're struggling. Please know:\n\n"
                "🤝 Talk to someone you trust\n"
                "💚 Be gentle with yourself\n"
                "🌱 Take things moment by moment\n\n"
                "**Need help now?**\n"
                "• 988 Lifeline\n"
                "• Crisis Text Line: HOME to 741741")

    elif mood == 'negative':
        return ("Having a tough time? That's okay:\n\n"
                "🌿 Take a short walk\n"
                "🎵 Listen to uplifting music\n"
                "📞 Connect with support\n\n"
                "This feeling is temporary. You've got this! 💪")

    elif mood in ['positive', 'very_positive']:
        return ("Love your positive energy! 🌟\n\n"
                "✨ Set growth intentions\n"
                "🤗 Share kindness\n"
                "📝 Document what's working\n\n"
                "Keep this momentum! 😊")

    return "Thank you for sharing. I'm here to support your wellness journey. How can I help you today? 💚"


# ==================== AUTHENTICATION ROUTES ====================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400

        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not username or not email or not password:
            return jsonify({'error': 'All fields required'}), 400

        # Validate username
        valid, msg = validate_username(username)
        if not valid:
            return jsonify({'error': msg}), 400

        # Validate password
        valid, msg = validate_password(password)
        if not valid:
            return jsonify({'error': msg}), 400

        # Validate email format
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            return jsonify({'error': 'Invalid email format'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user, remember=True)

        return jsonify({
            'message': 'Registration successful',
            'user': user.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Registration error: {e}')
        return jsonify({'error': 'Registration failed. Please try again.'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400

        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid credentials'}), 401

        login_user(user, remember=True)
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict()
        })

    except Exception as e:
        app.logger.error(f'Login error: {e}')
        return jsonify({'error': 'Login failed. Please try again.'}), 500


@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    """User logout"""
    logout_user()
    return jsonify({'message': 'Logout successful'})


@app.route('/api/auth/user', methods=['GET'])
@login_required
def get_current_user():
    """Get current user info"""
    return jsonify(current_user.to_dict())


# ==================== JOURNAL ROUTES ====================

@app.route('/api/journal/add', methods=['POST'])
@login_required
def add_journal():
    """Add journal entry"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400

        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        if len(text) > 10000:
            return jsonify({'error': 'Entry too long (max 10,000 characters)'}), 400

        sentiment = analyze_sentiment(text)
        emotions = extract_emotions(text)

        entry = JournalEntry(
            user_id=current_user.id,
            text=text,
            polarity=sentiment['polarity'],
            subjectivity=sentiment['subjectivity'],
            mood=sentiment['mood'],
            emotions=json.dumps(emotions)
        )
        db.session.add(entry)
        db.session.commit()

        return jsonify({
            'message': 'Entry saved',
            'entry': entry.to_dict(),
            'sentiment': sentiment,
            'emotions': emotions
        }), 201

    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Journal add error: {e}')
        return jsonify({'error': 'Failed to save entry'}), 500


@app.route('/api/journal/entries', methods=['GET'])
@login_required
def get_journal_entries():
    """Get all entries with pagination"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(per_page, 100)  # Cap at 100

    entries = JournalEntry.query.filter_by(user_id=current_user.id)\
        .order_by(JournalEntry.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    return jsonify([entry.to_dict() for entry in entries.items])


@app.route('/api/journal/search', methods=['GET'])
@login_required
def search_journal():
    """Search entries"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])

    entries = JournalEntry.query.filter(
        JournalEntry.user_id == current_user.id,
        JournalEntry.text.ilike(f'%{query}%')  # Case-insensitive search
    ).order_by(JournalEntry.created_at.desc()).limit(50).all()

    return jsonify([entry.to_dict() for entry in entries])


@app.route('/api/journal/delete/<int:entry_id>', methods=['DELETE'])
@login_required
def delete_journal_entry(entry_id):
    """Delete entry"""
    entry = JournalEntry.query.filter_by(id=entry_id, user_id=current_user.id).first()
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404

    db.session.delete(entry)
    db.session.commit()
    return jsonify({'message': 'Entry deleted'})


# ==================== MOOD ROUTES ====================

@app.route('/api/mood/log', methods=['POST'])
@login_required
def log_mood():
    """Quick mood check-in"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400

        mood = data.get('mood')
        if not mood:
            return jsonify({'error': 'Mood is required'}), 400

        energy = data.get('energy_level')
        stress = data.get('stress_level')

        # Validate integer fields
        if energy is not None and not (1 <= int(energy) <= 10):
            return jsonify({'error': 'Energy level must be between 1 and 10'}), 400
        if stress is not None and not (1 <= int(stress) <= 10):
            return jsonify({'error': 'Stress level must be between 1 and 10'}), 400

        mood_log = MoodLog(
            user_id=current_user.id,
            mood=mood,
            energy_level=energy,
            stress_level=stress,
            notes=data.get('notes', '')
        )
        db.session.add(mood_log)
        db.session.commit()

        return jsonify({'message': 'Mood logged', 'log': mood_log.to_dict()}), 201

    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Mood log error: {e}')
        return jsonify({'error': 'Failed to log mood'}), 500


@app.route('/api/mood/history', methods=['GET'])
@login_required
def get_mood_history():
    """Get mood history"""
    days = request.args.get('days', 30, type=int)
    days = min(days, 365)  # Cap at 1 year
    date_threshold = datetime.utcnow() - timedelta(days=days)

    entries = JournalEntry.query.filter(
        JournalEntry.user_id == current_user.id,
        JournalEntry.created_at >= date_threshold
    ).order_by(JournalEntry.created_at.asc()).all()

    history = [{
        'date': entry.created_at.strftime('%Y-%m-%d'),
        'mood': entry.mood,
        'polarity': entry.polarity,
        'type': 'journal'
    } for entry in entries]

    return jsonify(history)


@app.route('/api/mood/stats', methods=['GET'])
@login_required
def get_mood_stats():
    """Get mood statistics"""
    entries = JournalEntry.query.filter_by(user_id=current_user.id).all()

    if not entries:
        return jsonify({'total_entries': 0, 'message': 'No data yet'})

    moods = [e.mood for e in entries]
    polarities = [e.polarity for e in entries if e.polarity is not None]
    mood_counts = Counter(moods)

    if len(entries) >= 5:
        recent = entries[-3:]
        earlier = entries[:3]
        recent_avg = np.mean([e.polarity for e in recent if e.polarity is not None])
        earlier_avg = np.mean([e.polarity for e in earlier if e.polarity is not None])
        trend = 'improving' if recent_avg > earlier_avg + 0.1 else 'declining' if recent_avg < earlier_avg - 0.1 else 'stable'
    else:
        trend = 'insufficient_data'

    return jsonify({
        'total_entries': len(entries),
        'average_polarity': round(float(np.mean(polarities)), 3) if polarities else 0,
        'mood_distribution': dict(mood_counts),
        'trend': trend,
        'most_common_mood': mood_counts.most_common(1)[0][0] if mood_counts else None
    })


# ==================== GOAL ROUTES ====================

@app.route('/api/goals', methods=['GET'])
@login_required
def get_goals():
    """Get all goals"""
    goals = Goal.query.filter_by(user_id=current_user.id)\
        .order_by(Goal.created_at.desc()).all()
    return jsonify([goal.to_dict() for goal in goals])


@app.route('/api/goals/add', methods=['POST'])
@login_required
def add_goal():
    """Add new goal"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400

        title = data.get('title', '').strip()
        if not title:
            return jsonify({'error': 'Goal title is required'}), 400

        target_value = data.get('target_value')
        if target_value is None or int(target_value) <= 0:
            return jsonify({'error': 'Target value must be a positive number'}), 400

        deadline = None
        if data.get('deadline'):
            try:
                deadline = datetime.fromisoformat(data['deadline']).date()
            except ValueError:
                return jsonify({'error': 'Invalid deadline format'}), 400

        goal = Goal(
            user_id=current_user.id,
            title=title,
            description=data.get('description', ''),
            category=data.get('category', 'general'),
            target_value=int(target_value),
            unit=data.get('unit', ''),
            deadline=deadline
        )
        db.session.add(goal)
        db.session.commit()

        return jsonify({'message': 'Goal created', 'goal': goal.to_dict()}), 201

    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Goal add error: {e}')
        return jsonify({'error': 'Failed to create goal'}), 500


@app.route('/api/goals/<int:goal_id>/update', methods=['PATCH'])
@login_required
def update_goal_progress(goal_id):
    """Update goal progress"""
    goal = Goal.query.filter_by(id=goal_id, user_id=current_user.id).first()
    if not goal:
        return jsonify({'error': 'Goal not found'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request data'}), 400

    if 'current_value' in data:
        new_value = max(0, int(data['current_value']))  # Ensure non-negative
        goal.current_value = new_value
        if goal.target_value and goal.current_value >= goal.target_value:
            goal.completed = True

    db.session.commit()
    return jsonify(goal.to_dict())


@app.route('/api/goals/<int:goal_id>/complete', methods=['POST'])
@login_required
def complete_goal(goal_id):
    """Mark goal complete"""
    goal = Goal.query.filter_by(id=goal_id, user_id=current_user.id).first()
    if not goal:
        return jsonify({'error': 'Goal not found'}), 404

    goal.completed = True
    goal.current_value = goal.target_value  # Set to 100%
    db.session.commit()

    return jsonify({'message': 'Goal completed!', 'goal': goal.to_dict()})


@app.route('/api/goals/<int:goal_id>', methods=['DELETE'])
@login_required
def delete_goal(goal_id):
    """Delete a goal"""
    goal = Goal.query.filter_by(id=goal_id, user_id=current_user.id).first()
    if not goal:
        return jsonify({'error': 'Goal not found'}), 404

    db.session.delete(goal)
    db.session.commit()
    return jsonify({'message': 'Goal deleted'})


# ==================== CHAT ROUTES ====================

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """AI chatbot"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400

        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Message required'}), 400

        if len(user_message) > 2000:
            return jsonify({'error': 'Message too long (max 2000 characters)'}), 400

        sentiment = analyze_sentiment(user_message)
        emotions = extract_emotions(user_message)
        bot_response = generate_chatbot_response(user_message, sentiment)

        chat_msg = ChatMessage(
            user_id=current_user.id,
            user_message=user_message,
            bot_response=bot_response,
            sentiment_mood=sentiment['mood']
        )
        db.session.add(chat_msg)
        db.session.commit()

        return jsonify({
            'response': bot_response,
            'sentiment': sentiment,
            'emotions': emotions,
            'timestamp': chat_msg.created_at.isoformat()
        })

    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Chat error: {e}')
        return jsonify({'error': 'Chat service unavailable'}), 500


@app.route('/api/chat/history', methods=['GET'])
@login_required
def get_chat_history():
    """Get chat history"""
    limit = request.args.get('limit', 50, type=int)
    limit = min(limit, 100)

    messages = ChatMessage.query.filter_by(user_id=current_user.id)\
        .order_by(ChatMessage.created_at.desc()).limit(limit).all()

    return jsonify([msg.to_dict() for msg in reversed(messages)])


# ==================== MUSIC RECOMMENDATIONS ====================

MOOD_MUSIC_MAP = {
    'very_positive': {
        'playlists': [
            {'name': 'Happy Hits!', 'spotify_uri': '37i9dQZF1DXdPec7aLTmlC'},
            {'name': 'Good Vibes', 'spotify_uri': '37i9dQZF1DX0UrRvztWcAU'}
        ]
    },
    'positive': {
        'playlists': [
            {'name': 'Chill Hits', 'spotify_uri': '37i9dQZF1DX4WYpdgoIcn6'},
            {'name': 'Peaceful Piano', 'spotify_uri': '37i9dQZF1DX4sWSpwq3LiO'}
        ]
    },
    'neutral': {
        'playlists': [
            {'name': 'Lo-Fi Beats', 'spotify_uri': '37i9dQZF1DWWQRwui0ExPn'}
        ]
    },
    'negative': {
        'playlists': [
            {'name': 'Calm Vibes', 'spotify_uri': '37i9dQZF1DX3Ogo9pFvBkY'}
        ]
    },
    'very_negative': {
        'playlists': [
            {'name': 'Deep Focus', 'spotify_uri': '37i9dQZF1DWZeKCadgRdKQ'}
        ]
    }
}

VALID_MOODS = {'very_positive', 'positive', 'neutral', 'negative', 'very_negative'}

@app.route('/api/recommendations/music', methods=['POST'])
def get_music_recommendations():
    """Get music recommendations"""
    data = request.get_json() or {}
    mood = data.get('mood', 'neutral')
    if mood not in VALID_MOODS:
        mood = 'neutral'
    return jsonify(MOOD_MUSIC_MAP[mood])


@app.route('/api/recommendations/videos', methods=['POST'])
def get_video_recommendations():
    """Get video recommendations"""
    data = request.get_json() or {}
    mood = data.get('mood', 'neutral')
    if mood not in VALID_MOODS:
        mood = 'neutral'

    video_map = {
        'very_positive': ['gratitude meditation', 'celebrate wins'],
        'positive': ['mindfulness practice', 'morning yoga'],
        'neutral': ['guided meditation', 'breathing exercises'],
        'negative': ['stress relief', 'calming music'],
        'very_negative': ['deep relaxation', 'anxiety relief']
    }
    return jsonify({'search_terms': video_map[mood], 'mood': mood})


# ==================== SETTINGS ====================

@app.route('/api/settings/theme', methods=['POST'])
@login_required
def update_theme():
    """Update theme"""
    data = request.get_json() or {}
    theme = data.get('theme', 'light')
    if theme not in ('light', 'dark'):
        return jsonify({'error': 'Invalid theme'}), 400

    current_user.theme = theme
    db.session.commit()
    return jsonify({'message': 'Theme updated', 'theme': theme})


# ==================== STATIC FILES ====================

@app.route('/')
def index():
    """Serve main app"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/login.html')
def login_page():
    """Serve login page"""
    return send_from_directory(app.static_folder, 'login.html')


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(e):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500


# ==================== DATABASE INIT ====================

def init_db():
    """Initialize database"""
    with app.app_context():
        db.create_all()
        print("✅ Database initialized successfully")


# ==================== MAIN ====================

if __name__ == '__main__':
    init_db()
    print("=" * 60)
    print("🎧 HearMe - Mental Wellness AI Companion")
    print("=" * 60)
    print("Server: http://localhost:5000")
    print("\n✅ Features Active:")
    print("  • User Authentication (with validation)")
    print("  • Database Storage (SQLite)")
    print("  • Journal & Sentiment Analysis")
    print("  • Mood Tracking")
    print("  • Goal Setting & Tracking")
    print("  • AI Chatbot")
    print("  • Music & Video Recommendations")
    print("  • Input Validation & Error Handling")
    print("  • Pagination Support")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5000)