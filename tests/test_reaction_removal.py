import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, mock_open, patch

from pyrogram import raw, types

os.environ.setdefault('API_ID', '12345')
os.environ.setdefault('API_HASH', 'test-api-hash')

with (
    patch('os.path.exists', return_value=False),
    patch('builtins.open', mock_open()),
    patch('pyrogram.Client.run'),
):
    import cleaner as cleaner_module

from cleaner import Cleaner, parse_args


def reaction(*, chosen_order=None, kind='emoji'):
    return SimpleNamespace(
        chosen_order=chosen_order,
        type=SimpleNamespace(type=kind),
    )


def message(message_id, *reactions):
    summary = SimpleNamespace(reactions=list(reactions)) if reactions else None
    return SimpleNamespace(id=message_id, reactions=summary)


class FakeApp:
    def __init__(self, histories, invoke_effects=None):
        self.histories = histories
        self.invoke_effects = list(invoke_effects or [])
        self.resolved = []
        self.requests = []

    def get_chat_history(self, chat_id):
        async def history():
            for item in self.histories[chat_id]:
                yield item

        return history()

    async def resolve_peer(self, chat_id):
        self.resolved.append(chat_id)
        return raw.types.InputPeerSelf()

    async def invoke(self, request):
        self.requests.append(request)
        if self.invoke_effects:
            effect = self.invoke_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
        return True


class ReactionRemovalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.chat = SimpleNamespace(id=-100123, title='Test group', username=None)
        self.cleaner = Cleaner(chats=[self.chat])

    async def test_group_picker_describes_reaction_mode(self):
        self.cleaner.get_groups = AsyncMock(return_value=([self.chat], {}))
        output = StringIO()

        with patch('builtins.input', return_value='1'), redirect_stdout(output):
            await self.cleaner.select_groups(remove_reactions=True)

        self.assertIn('Remove all your reactions from messages in', output.getvalue())
        self.assertNotIn('Delete all your messages in', output.getvalue())

    async def test_wrong_confirmation_makes_no_mutation_calls(self):
        fake_app = FakeApp({self.chat.id: [message(10, reaction(chosen_order=0))]})
        output = StringIO()

        with (
            patch.object(cleaner_module, 'app', fake_app),
            patch('builtins.input', return_value='remove reactions'),
            redirect_stdout(output),
        ):
            await self.cleaner.run(remove_reactions=True)

        self.assertEqual([], fake_app.requests)
        self.assertIn('Reaction removal preview:', output.getvalue())
        self.assertIn('Confirmation did not match', output.getvalue())

    async def test_eof_makes_no_mutation_calls(self):
        fake_app = FakeApp({self.chat.id: [message(10, reaction(chosen_order=0))]})

        with (
            patch.object(cleaner_module, 'app', fake_app),
            patch('builtins.input', side_effect=EOFError),
            redirect_stdout(StringIO()),
        ):
            await self.cleaner.run(remove_reactions=True)

        self.assertEqual([], fake_app.requests)

    async def test_exact_confirmation_retracts_only_own_non_paid_reactions(self):
        fake_app = FakeApp({
            self.chat.id: [
                message(10, reaction(chosen_order=0)),
                message(20, reaction(chosen_order=None)),
                message(30, reaction(chosen_order=1, kind='paid')),
                message(40),
                message(
                    50,
                    reaction(chosen_order=2, kind='paid'),
                    reaction(chosen_order=1),
                ),
            ]
        })
        output = StringIO()

        with (
            patch.object(cleaner_module, 'app', fake_app),
            patch('builtins.input', return_value='REMOVE REACTIONS'),
            redirect_stdout(output),
        ):
            await self.cleaner.run(remove_reactions=True)

        self.assertEqual([self.chat.id], fake_app.resolved)
        self.assertEqual([10, 50], [request.msg_id for request in fake_app.requests])
        self.assertTrue(all(request.reaction is None for request in fake_app.requests))
        serialized_flags = int.from_bytes(fake_app.requests[0].write()[4:8], 'little')
        self.assertEqual(0, serialized_flags & 1)
        self.assertIn('2 messages with removable reactions', output.getvalue())
        self.assertIn('2 messages also contain paid reactions', output.getvalue())

    async def test_pinned_client_parser_preserves_chosen_order_zero(self):
        raw_summary = raw.types.MessageReactions(
            results=[
                raw.types.ReactionCount(
                    reaction=raw.types.ReactionEmoji(emoticon='👍'),
                    count=1,
                    chosen_order=0,
                )
            ]
        )
        parsed_summary = types.MessageReactions._parse(None, raw_summary)
        fake_app = FakeApp({
            self.chat.id: [SimpleNamespace(id=10, reactions=parsed_summary)],
        })

        with patch.object(cleaner_module, 'app', fake_app), redirect_stdout(StringIO()):
            message_ids, paid_count = await self.cleaner.search_reactions(self.chat.id)

        self.assertEqual([10], message_ids)
        self.assertEqual(0, paid_count)

    async def test_empty_inventory_does_not_prompt_or_resolve_peer(self):
        fake_app = FakeApp({
            self.chat.id: [
                message(10, reaction(chosen_order=None)),
                message(20, reaction(chosen_order=0, kind='paid')),
            ]
        })

        with (
            patch.object(cleaner_module, 'app', fake_app),
            patch('builtins.input', side_effect=AssertionError('must not prompt')),
            redirect_stdout(StringIO()),
        ):
            await self.cleaner.run(remove_reactions=True)

        self.assertEqual([], fake_app.resolved)
        self.assertEqual([], fake_app.requests)

    async def test_flood_wait_retries_the_same_message(self):
        class FakeFloodWait(Exception):
            def __init__(self, value):
                super().__init__(value)
                self.value = value

        fake_app = FakeApp(
            {self.chat.id: [message(10, reaction(chosen_order=0))]},
            invoke_effects=[FakeFloodWait(0), True],
        )

        with (
            patch.object(cleaner_module, 'app', fake_app),
            patch.object(cleaner_module, 'FloodWait', FakeFloodWait),
            patch.object(cleaner_module, 'sleep', new_callable=AsyncMock) as mocked_sleep,
            patch('builtins.input', return_value='REMOVE REACTIONS'),
            redirect_stdout(StringIO()),
        ):
            await self.cleaner.run(remove_reactions=True)

        self.assertEqual([10, 10], [request.msg_id for request in fake_app.requests])
        mocked_sleep.assert_awaited_once_with(0)

    async def test_inventory_failure_happens_before_any_mutation(self):
        second_chat = SimpleNamespace(id=-100456, title='Broken group')

        class FailingInventoryApp(FakeApp):
            def get_chat_history(self, chat_id):
                if chat_id == second_chat.id:
                    async def failing_history():
                        raise RuntimeError('history unavailable')
                        yield

                    return failing_history()
                return super().get_chat_history(chat_id)

        fake_app = FailingInventoryApp({
            self.chat.id: [message(10, reaction(chosen_order=0))],
        })
        cleaner = Cleaner(chats=[self.chat, second_chat])

        with (
            patch.object(cleaner_module, 'app', fake_app),
            patch('builtins.input', side_effect=AssertionError('must not prompt')),
            redirect_stdout(StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, 'history unavailable'):
                await cleaner.run(remove_reactions=True)

        self.assertEqual([], fake_app.requests)

    async def test_mutation_failure_is_not_reported_as_complete(self):
        fake_app = FakeApp(
            {
                self.chat.id: [
                    message(10, reaction(chosen_order=0)),
                    message(20, reaction(chosen_order=1)),
                ]
            },
            invoke_effects=[True, RuntimeError('request failed')],
        )
        output = StringIO()

        with (
            patch.object(cleaner_module, 'app', fake_app),
            patch('builtins.input', return_value='REMOVE REACTIONS'),
            redirect_stdout(output),
        ):
            with self.assertRaisesRegex(RuntimeError, 'request failed'):
                await self.cleaner.run(remove_reactions=True)

        self.assertEqual([10, 20], [request.msg_id for request in fake_app.requests])
        self.assertNotIn('Removed reactions from 2 messages.', output.getvalue())

    async def test_default_run_uses_message_path_not_reaction_path(self):
        self.cleaner.search_messages = AsyncMock(return_value=[SimpleNamespace(id=10)])
        self.cleaner.delete_messages = AsyncMock()
        self.cleaner.run_reaction_removal = AsyncMock()

        with redirect_stdout(StringIO()):
            await self.cleaner.run()

        self.cleaner.search_messages.assert_awaited()
        self.cleaner.run_reaction_removal.assert_not_awaited()


class ReactionArgumentTests(unittest.TestCase):
    def test_default_mode_removes_messages(self):
        with patch.object(sys, 'argv', ['cleaner.py']):
            args = parse_args()

        self.assertFalse(args.remove_reactions)

    def test_full_reaction_option_selects_reaction_mode(self):
        with patch.object(sys, 'argv', ['cleaner.py', '--remove-reactions']):
            args = parse_args()

        self.assertTrue(args.remove_reactions)

    def test_reaction_option_cannot_be_abbreviated(self):
        with patch.object(sys, 'argv', ['cleaner.py', '--remove-r']):
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as exit_context:
                    parse_args()

        self.assertEqual(2, exit_context.exception.code)

    @unittest.skipUnless(os.name == 'nt', 'Windows batch launcher test')
    def test_windows_launcher_forwards_reaction_argument(self):
        project_root = os.path.dirname(os.path.dirname(__file__))

        with tempfile.TemporaryDirectory() as directory:
            launcher = os.path.join(directory, 'start.bat')
            shutil.copy2(os.path.join(project_root, 'start.bat'), launcher)

            scripts = os.path.join(directory, 'venv', 'Scripts')
            os.makedirs(scripts)
            with open(os.path.join(scripts, 'activate.bat'), 'w') as activate:
                activate.write('@set PATH=%~dp0;%PATH%\n')
            with open(os.path.join(scripts, 'python.bat'), 'w') as python:
                python.write('@echo %*>"%CAPTURE_FILE%"\n')
            with open(os.path.join(scripts, 'deactivate.bat'), 'w') as deactivate:
                deactivate.write('@echo off\n')

            capture_file = os.path.join(directory, 'arguments.txt')
            environment = os.environ.copy()
            environment['CAPTURE_FILE'] = capture_file
            subprocess.run(
                ['cmd', '/c', launcher, '--remove-reactions'],
                check=True,
                capture_output=True,
                env=environment,
                text=True,
            )

            with open(capture_file, 'r') as captured:
                arguments = captured.read().strip()
            self.assertTrue(
                arguments.endswith('cleaner.py --remove-reactions'),
                arguments,
            )


if __name__ == '__main__':
    unittest.main()
